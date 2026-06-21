"""Prototype: lengthscale-matched subsampling + 'smoothing improves W'.
Run to verify the story before packaging into a notebook."""
import sys, time, numpy as np, torch, torch.nn as nn, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/home/willwhite/GitHub/gllvm/src")
from gllvm.autofit import procrustes_error
from collections import namedtuple
torch.set_default_dtype(torch.float64)

Q, P, WZS = 2, 30, 0.7
DT = 1.0
ELL_TRUE = [2.0, 8.0]          # distinct per-latent timescales (moderate smoothness)
JIT = 1e-4
Params = namedtuple("Params", ["W", "b", "ells"])

# ------------------------------------------------------------------ core (B=I, batched over arbitrary coords)
def vec(Z):  return Z.transpose(1, 2).reshape(Z.shape[0], -1)          # (n,K,Q)->(n,QK)
def unvec(u, K): return u.reshape(u.shape[0], Q, K).transpose(1, 2)     # (n,QK)->(n,K,Q)

def Sigma_b(ts, ells):                       # ts:(n,K) -> (n,QK,QK) block-diagonal (B=I)
    n, K = ts.shape
    d2 = (ts[:, :, None] - ts[:, None, :]) ** 2
    S = ts.new_zeros(n, Q * K, Q * K)
    for k in range(Q):
        S[:, k*K:(k+1)*K, k*K:(k+1)*K] = torch.exp(-0.5 * d2 / ells[k] ** 2)
    return S + JIT * torch.eye(Q * K)

def chol_b(ts, ells): return torch.linalg.cholesky(Sigma_b(ts, ells))

def sample_y_b(eps, ts, p):                  # eps:(n,K,Q)
    n, K, _ = eps.shape
    L = chol_b(ts, p.ells)
    z = unvec((L @ vec(eps).unsqueeze(-1)).squeeze(-1), K)            # z_vec = L eps
    eta = torch.einsum("nkq,pq->nkp", z, p.W) + p.b
    return torch.poisson(torch.exp(eta.clamp(max=10)))

def encode_b(y, ts, p, s2=1.0):              # y:(n,K,P) -> zhat (n,K,Q)  block-MAP imputer
    n, K, _ = y.shape
    L = chol_b(ts, p.ells)
    Lit = torch.linalg.solve_triangular(L, torch.eye(Q*K).expand(n, -1, -1), upper=False)
    A = Lit.transpose(1, 2) @ Lit + torch.kron(p.W.T @ p.W / s2, torch.eye(K))
    rhs = vec((torch.log1p(y) - p.b) @ p.W / s2)
    LA = torch.linalg.cholesky(A)
    return unvec(torch.cholesky_solve(rhs.unsqueeze(-1), LA).squeeze(-1), K)

# ------------------------------------------------------------------ subsampling
def matched_offsets(K):
    """ℓ-matched = contiguous local window of K neighbours (random anchor).
    Width (K-1)*dt scales with how much correlation we pool; pairwise distances
    span every scale dt..(K-1)*dt, so it also identifies distinct ℓ_k."""
    return torch.arange(K) - K // 2

def sub_matched(Yf, offs, batch, dt):
    BG, G, _ = Yf.shape; m = int(offs.abs().max())
    g = torch.randint(0, BG, (batch,)); a = torch.randint(m, G - m, (batch,))
    idx = a[:, None] + offs[None, :]                                  # (batch,K) sorted
    return Yf[g[:, None], idx], idx.double() * dt

def sub_full(Yf, batch, dt):                  # whole block of every group (no subsampling)
    BG, G, _ = Yf.shape
    g = torch.randint(0, BG, (batch,))
    idx = torch.arange(G).expand(batch, G)
    return Yf[g[:, None], idx], idx.double() * dt

# ------------------------------------------------------------------ data + fitter
def gen_data(BG, G, b0, seed):
    torch.manual_seed(1000 + seed)
    W = torch.randn(P, Q) * WZS; b = torch.full((P,), float(b0))
    ells = torch.tensor(ELL_TRUE)
    ts_full = (torch.arange(G).double() * DT).expand(BG, G)
    eps = torch.randn(BG, G, Q)
    Yf = sample_y_b(eps, ts_full, Params(W, b, ells))
    return Yf, W, b

def fit(Yf, sampler, fit_ell=True, ell_fixed=None, ell_init=(1.5, 3.5),
        steps=500, lr=0.03, batch=96, warm=150, seed=0):
    torch.manual_seed(7000 + seed)
    W = nn.Parameter(torch.randn(P, Q) * WZS); b = nn.Parameter(torch.zeros(P))
    if fit_ell:
        log_ell = nn.Parameter(torch.log(torch.tensor(ell_init))); popt = [W, b, log_ell]
    else:
        log_ell = torch.log(torch.tensor(ell_fixed)); popt = [W, b]
    opt = torch.optim.Adam(popt, lr=lr)
    for it in range(steps):
        ells = log_ell.exp(); p = Params(W, b, ells)
        yb, ts = sampler()
        with torch.no_grad():
            pdet = Params(W.detach(), b.detach(), ells.detach())
            yq = sample_y_b(torch.randn(yb.shape[0], ts.shape[1], Q), ts, pdet)
            zd = encode_b(yb, ts, pdet); zq = encode_b(yq, ts, pdet)
        ed = torch.einsum("nkq,pq->nkp", zd, W) + b
        eq = torch.einsum("nkq,pq->nkp", zq, W) + b
        loss = -((torch.log1p(yb) * ed).sum(-1).mean() - (torch.log1p(yq) * eq).sum(-1).mean())
        if fit_ell:                                # second-moment ℓ fit (shared stencil coords)
            with torch.no_grad():
                Md = torch.einsum("ni,nj->ij", vec(zd), vec(zd)) / yb.shape[0]
                Mq = torch.einsum("ni,nj->ij", vec(zq), vec(zq)) / yb.shape[0]
                Sig_model = Sigma_b(ts[:1], ells)[0]
                Sig_hat = Sig_model + (Md - Mq)
            loss = loss + ((Sigma_b(ts[:1], ells)[0] - Sig_hat) ** 2).mean()
        opt.zero_grad(); loss.backward()
        if it < warm and fit_ell: log_ell.grad = None
        torch.nn.utils.clip_grad_norm_(popt, 5.0); opt.step()
    ell_out = sorted(log_ell.exp().tolist()) if fit_ell else list(ell_fixed)
    return W.detach(), ell_out

NSEED, BG, GFULL, KSUB = 8, 16, 64, 25
SPARS = [-2.5, -1.5, -0.5, 0.5]
STEPS, EINIT = 600, (1.5, 5.0)
offs = matched_offsets(KSUB)
t0 = time.time()
# TRUTH = GP-GLLVM (ℓ_true).  Three fits on the SAME data per (sparsity, seed):
#   GPGLLVM_full : Fit 1 (GP-GLLVM), whole block            -> parity reference
#   GPGLLVM_sub  : Fit 1 (GP-GLLVM), K-window subsample      -> Panel 1: sub ≈ full
#   GLLVM_sub    : Fit 2 (standard GLLVM, independent), sub  -> Panel 2: worse than Fit 1
print("=" * 80)
print(f"TRUTH = GP-GLLVM (ℓ_true={ELL_TRUE}). full G={GFULL} | sub K={KSUB} window | {NSEED} seeds")
print(f"{'mean count':>10} | {'GP-GLLVM full':>13} {'GP-GLLVM sub':>13} | {'standard GLLVM':>14} | {'ℓ̂ (GP-GLLVM sub)':>18}")
rows = []
for b0 in SPARS:
    mc, pf, ps, pi, eh = [], [], [], [], []
    for s in range(NSEED):
        Yf, Wt, _ = gen_data(BG=BG, G=GFULL, b0=b0, seed=s)
        mc.append(Yf.double().mean().item())
        smp_full = lambda: sub_full(Yf, BG, DT)
        smp_sub  = lambda: sub_matched(Yf, offs, 96, DT)
        pf.append(procrustes_error(Wt, fit(Yf, smp_full, fit_ell=True,  ell_init=EINIT, steps=STEPS, batch=BG, seed=s)[0]))
        Ws, e = fit(Yf, smp_sub, fit_ell=True, ell_init=EINIT, steps=STEPS, seed=s)
        ps.append(procrustes_error(Wt, Ws)); eh.append(e)
        pi.append(procrustes_error(Wt, fit(Yf, smp_sub, fit_ell=False, ell_fixed=(1e-3, 1e-3), steps=STEPS, seed=s)[0]))
    mc, pf, ps, pi = np.mean(mc), np.mean(pf), np.mean(ps), np.mean(pi); eh = np.mean(eh, 0)
    rows.append((mc, pf, ps, pi))
    print(f"{mc:>10.2f} | {pf:>13.3f} {ps:>13.3f} | {pi:>14.3f} | [{eh[0]:.2f}, {eh[1]:.2f}]")
print(f"[total {time.time()-t0:.0f}s]")

# ------------------------------------------------------------------ paper figure
r = np.array(rows)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(r[:, 0], r[:, 1], "o-", label=f"GP-GLLVM, full block (G={GFULL})")
ax[0].plot(r[:, 0], r[:, 2], "s--", label=f"GP-GLLVM, subsample (K={KSUB})")
ax[0].set_title("1) subsampling costs ~nothing for W"); ax[0].legend()
ax[1].plot(r[:, 0], r[:, 2], "s-", label="Fit 1: GP-GLLVM (correct)")
ax[1].plot(r[:, 0], r[:, 3], "^-", label="Fit 2: standard GLLVM (independent)")
ax[1].set_title("2) modelling the GP recovers W better"); ax[1].legend()
for a in ax:
    a.set_xlabel("mean count / obs  (← sparser)"); a.set_ylabel("procrustes error of W"); a.invert_xaxis()
fig.suptitle(f"Truth = GP-GLLVM (ℓ={ELL_TRUE});  both fits on the same data", fontsize=11)
fig.tight_layout(); fig.savefig("/home/willwhite/GitHub/gllvm/playground/gp-gllvm/_subsample_demo.png", dpi=110)
print("saved _subsample_demo.png")
