"""GP-GLLVM scaling figure for the paper: wall-clock fit time is flat in the number
of observations n (K-subset / GP-marginal estimator), while accuracy improves with n.
Saves paper/figures/scaling.png."""
import sys, time, numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/home/willwhite/GitHub/gllvm/src")
from gllvm import GPGLLVM, GPZQEFitter, PoissonGLM, procrustes_error
torch.set_default_dtype(torch.float64)

Q, P, G, K = 2, 30, 80, 20            # latent dim, responses, group size, sub-sample size
ELL = [1.5, 3.0]
B_GRID = [25, 100, 400, 1600, 6400]   # number of groups  ->  n = B*G  (2e3 .. 5.1e5)
STEPS = 200

# true model (shared grid t per group -> one Cholesky, generate any number of groups cheaply)
torch.manual_seed(0)
gt = GPGLLVM(Q, P, input_dim=1, lengthscale=ELL)
gt.add_glm(PoissonGLM, idx=range(P), params={"T": torch.log1p}, name="P")
with torch.no_grad():
    gt.wz.normal_(0, 0.7); gt.bias.fill_(0.3)
t = (torch.arange(G).double())[:, None]                       # (G,1)
L = gt.kernel.cholesky(t[None])[0]                            # (q,G,G), one cholesky

def gen(B, seed):
    torch.manual_seed(1000 + seed)
    eps = torch.randn(B, G, Q)
    z = torch.einsum("qij,bqj->bqi", L, eps.transpose(1, 2)).transpose(1, 2)   # (B,G,q)
    eta = torch.einsum("bgq,pq->bgp", z, gt.wz) + gt.bias
    y = torch.poisson(torch.exp(eta.clamp(max=12)))
    return (y.reshape(B * G, P), t.repeat(B, 1),
            torch.arange(B).repeat_interleave(G))

print(f"{'n':>9} {'B':>6} {'fit_s':>7} {'procW':>7}")
ns, times, procs = [], [], []
for B in B_GRID:
    Y, coords, groups = gen(B, seed=B)
    gfit = GPGLLVM(Q, P, input_dim=1, lengthscale=2.0)
    gfit.add_glm(PoissonGLM, idx=range(P), params={"T": torch.log1p}, name="P")
    with torch.no_grad():
        gfit.wz.normal_(0, 0.7)
    t0 = time.perf_counter()
    ft = GPZQEFitter(gfit, K=K, steps=STEPS, batch=64, warmup=60, lr=0.03,
                     device="cpu", seed=0).fit(Y, coords, groups=groups)
    dt = time.perf_counter() - t0
    pw = procrustes_error(gt.wz.detach(), gfit.wz.detach())
    ns.append(B * G); times.append(dt); procs.append(pw)
    print(f"{B*G:>9d} {B:>6d} {dt:>7.1f} {pw:>7.3f}", flush=True)

ns = np.array(ns); times = np.array(times); procs = np.array(procs)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(ns, times, "o-", lw=2, label=f"$Z_q$ GP-GLLVM ($K={K}$ subset)")
ax[0].plot(ns, times[0] * ns / ns[0], "k--", lw=1, label="$O(n)$ reference")
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("number of observations $n$"); ax[0].set_ylabel("wall-clock fit time (s)")
ax[0].set_title("Fit time is flat in $n$"); ax[0].legend(); ax[0].grid(alpha=.3, which="both")
ax[1].plot(ns, procs, "s-", lw=2, color="#55A868")
ax[1].set_xscale("log"); ax[1].set_xlabel("number of observations $n$")
ax[1].set_ylabel("Procrustes error of $W$")
ax[1].set_title("Accuracy improves with $n$"); ax[1].grid(alpha=.3, which="both")
fig.tight_layout()
out = "/home/willwhite/GitHub/gllvm/paper/figures/scaling.png"
fig.savefig(out, dpi=130, bbox_inches="tight"); print("saved", out)
