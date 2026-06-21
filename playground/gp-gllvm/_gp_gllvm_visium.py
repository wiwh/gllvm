"""GP-GLLVM (ZQE) on real Visium spatial transcriptomics.
Latent z over 2D space, GP-correlated (per-factor isotropic lengthscale, B=I);
y_gene ~ Poisson(exp(W z + b + offset)).  Likelihood-free ZQE, subsampled local
patches (scales: every op is K x K, never n_spots x n_spots).
Outputs: per-factor spatial lengthscales (µm) + factor maps over the tissue."""
import sys, time, numpy as np, torch, torch.nn as nn, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
torch.set_default_dtype(torch.float64)

Q, K, MPOOL, JIT = 5, 80, 600, 1e-4
d = np.load("/home/willwhite/GitHub/gllvm/playground/gp-gllvm/_visium.npz", allow_pickle=True)
Y = torch.tensor(d["Y"]); XY = torch.tensor(d["xy"]); OFF = torch.tensor(d["offset"]); pitch = float(d["pitch"])
N, P = Y.shape
tree = cKDTree(d["xy"])
idx_dense = torch.tensor(tree.query(d["xy"], k=64)[1])             # dense KNN (for imputation), col 0 = self
idx_pool  = torch.tensor(tree.query(d["xy"], k=MPOOL)[1])         # wide pool: K drawn sparsely from these
pool_R = float(np.sqrt(MPOOL / np.pi))
print(f"Visium GP-GLLVM: N={N} spots, P={P} genes, Q={Q} factors; fit patch K={K} drawn from {MPOOL}-NN (radius≈{pool_R:.0f} units)")

# ------------------------------------------------------------------ 2D core (offset-aware, batched)
def vec(Z):  return Z.transpose(1, 2).reshape(Z.shape[0], -1)
def unvec(u, k): return u.reshape(u.shape[0], Q, k).transpose(1, 2)
def Sigma_b(c, ells):                          # c:(n,k,2) -> (n,Qk,Qk) block-diag (B=I)
    n, k = c.shape[:2]
    d2 = ((c[:, :, None, :] - c[:, None, :, :]) ** 2).sum(-1)
    S = c.new_zeros(n, Q * k, Q * k)
    for q in range(Q):
        S[:, q*k:(q+1)*k, q*k:(q+1)*k] = torch.exp(-0.5 * d2 / ells[q] ** 2)
    return S + JIT * torch.eye(Q * k)
def sample_y_b(eps, c, off, W, b, ells):       # eps:(n,k,Q), off:(n,k)
    n, k, _ = eps.shape
    z = unvec((torch.linalg.cholesky(Sigma_b(c, ells)) @ vec(eps).unsqueeze(-1)).squeeze(-1), k)
    eta = torch.einsum("nkq,pq->nkp", z, W) + b + off[..., None]
    return torch.poisson(torch.exp(eta.clamp(max=12)))
def encode_b(y, c, off, W, b, ells, s2=1.0):   # -> zhat (n,k,Q)
    n, k, _ = y.shape
    L = torch.linalg.cholesky(Sigma_b(c, ells))
    Lit = torch.linalg.solve_triangular(L, torch.eye(Q*k).expand(n, -1, -1), upper=False)
    A = Lit.transpose(1, 2) @ Lit + torch.kron(W.T @ W / s2, torch.eye(k))
    rhs = vec((torch.log1p(y) - b - off[..., None]) @ W / s2)
    return unvec(torch.cholesky_solve(rhs.unsqueeze(-1), torch.linalg.cholesky(A)).squeeze(-1), k)

def patch(batch):                              # random centres -> K spots drawn sparsely from a wide pool
    pool = idx_pool[torch.randint(0, N, (batch,))]                 # (batch,MPOOL)
    sel = torch.rand(batch, MPOOL).argsort(1)[:, :K]              # K of the pool (spans the scale)
    ii = pool.gather(1, sel)                                       # (batch,K)
    return Y[ii], XY[ii], OFF[ii]

# ------------------------------------------------------------------ fit
def fit(steps=700, lr=0.03, batch=48, warm=200, fit_ell=True, ell_fixed=None, seed=0):
    torch.manual_seed(seed)
    W = nn.Parameter(torch.randn(P, Q) * 0.3); b = nn.Parameter(torch.log1p(Y.mean(0)))
    if fit_ell:
        log_ell = nn.Parameter(torch.log(torch.linspace(1.0, 12.0, Q))); popt = [W, b, log_ell]
    else:
        log_ell = torch.log(torch.tensor(ell_fixed)); popt = [W, b]
    opt = torch.optim.Adam(popt, lr=lr); hist = []
    for it in range(steps):
        ells = log_ell.exp()
        yb, cb, ob = patch(batch)
        with torch.no_grad():
            Wd, bd, eld = W.detach(), b.detach(), ells.detach()
            yq = sample_y_b(torch.randn(batch, K, Q), cb, ob, Wd, bd, eld)
            zd = encode_b(yb, cb, ob, Wd, bd, eld); zq = encode_b(yq, cb, ob, Wd, bd, eld)
        ed = torch.einsum("nkq,pq->nkp", zd, W) + b + ob[..., None]
        eq = torch.einsum("nkq,pq->nkp", zq, W) + b + ob[..., None]
        loss = -((torch.log1p(yb) * ed).sum(-1).mean() - (torch.log1p(yq) * eq).sum(-1).mean())
        if fit_ell:
            with torch.no_grad():
                Md = torch.einsum("ni,nj->ij", vec(zd), vec(zd)) / batch
                Mq = torch.einsum("ni,nj->ij", vec(zq), vec(zq)) / batch
                Sig_hat = Sigma_b(cb[:1], ells)[0] + (Md - Mq)
            loss = loss + ((Sigma_b(cb[:1], ells)[0] - Sig_hat) ** 2).mean()
        opt.zero_grad(); loss.backward()
        if it < warm and fit_ell: log_ell.grad = None
        torch.nn.utils.clip_grad_norm_(popt, 5.0); opt.step()
        if it % 50 == 0: hist.append((it, *sorted(log_ell.exp().tolist())))
    return W.detach(), b.detach(), log_ell.exp().detach(), hist

t0 = time.time()
W, b, ells, hist = fit()
fit_t = time.time() - t0
print(f"fit {fit_t:.0f}s   lengthscales (µm) = {[f'{e*pitch*100/137:.0f}' for e in ells.tolist()]}")
print(f"  (in spot-pitch units: {[f'{e:.2f}' for e in ells.tolist()]}; fit-pool radius ≈ {pool_R:.0f} units)")

# impute a z-map: encode every spot's DENSE patch, keep the centre (col 0) — chunked, scalable
zmap = torch.empty(N, Q)
for s in range(0, N, 256):
    ii = idx_dense[s:s+256]
    zmap[s:s+256] = encode_b(Y[ii], XY[ii], OFF[ii], W, b, ells)[:, 0, :]

# ------------------------------------------------------------------ figure: factor maps
xy = d["xy"]; order = np.argsort(ells.numpy())     # fine -> coarse
fig, ax = plt.subplots(1, Q, figsize=(3.1 * Q, 3.4))
for j, q in enumerate(order):
    z = zmap[:, q].numpy(); v = np.percentile(np.abs(z), 98)
    ax[j].scatter(xy[:, 0], xy[:, 1], c=z, s=6, cmap="RdBu_r", vmin=-v, vmax=v)
    ax[j].set_title(f"factor {q}  ℓ={ells[q]*100*pitch/137:.0f}µm"); ax[j].set_aspect("equal"); ax[j].axis("off")
    ax[j].invert_yaxis()
fig.suptitle(f"GP-GLLVM on Visium mouse brain ({N} spots, {P} genes) — fit {fit_t:.0f}s on local K={K} patches", y=1.02)
fig.tight_layout(); fig.savefig("/home/willwhite/GitHub/gllvm/playground/gp-gllvm/_visium_factors.png", dpi=110, bbox_inches="tight")
print("saved _visium_factors.png   ell-trace:", hist[-1])
