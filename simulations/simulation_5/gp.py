"""
gp.py — self-contained GP-GLLVM + composite-ZQE estimator for *simulation_5*.

A longitudinal/grouped GP-GLLVM: B groups, each with K observations on a shared
input grid ``t`` (e.g. timepoints). Within a group the latent factors are a
Gaussian process; groups are independent. Poisson (log-link) responses.

Generative (per group): ε ~ N(0,I_{K×q}); z = L(ℓ)ε (L=chol K(t)); y ~ Poisson(e^{Wz+b}).

Composite ZQE estimator (recovers W, b, lengthscale): the kernel lives in the
DECODER (ε→z=L(ℓ)ε→η), so ℓ is in η and gets gradients by backprop; the encoder is
no-grad imputation (per-obs MAP → whiten ε̂=L(ℓ)⁻¹Ẑ). Plain centered ZQE −(m1−m2),
**block-minibatched**: each step samples a few blocks and draws fantasy blocks — so
per-step cost is O(batch·K·p + K³), INDEPENDENT of the total number of groups B.
This is the engine that fits GP-GLLVMs at n = B·K in the millions, where the full
GP likelihood (O(n³)) is impossible. (Block estimation for GPs is the
Vecchia/composite-likelihood idea; the contribution here is its likelihood-free
realization for non-Gaussian GLLVMs via ZQE.)
"""
from __future__ import annotations
import time
import numpy as np
import torch


def rbf_kernel(t, lengthscale, variance=1.0, jitter=1e-3):
    t = torch.as_tensor(t, dtype=torch.float64).reshape(-1)
    d2 = (t[:, None] - t[None, :]) ** 2
    K = variance * torch.exp(-0.5 * d2 / (lengthscale ** 2))
    return K + jitter * torch.eye(len(t), dtype=K.dtype, device=K.device)


class GPGLLVM:
    """Grouped GP-GLLVM (Poisson). Parameters W (p,q), b (p,), kernel (ℓ, var)."""
    def __init__(self, q, p, lengthscale=2.0, kernel_var=1.0, wz_scale=0.7, seed=0, device="cpu"):
        g = torch.Generator().manual_seed(seed)
        self.q, self.p, self.kernel_var = q, p, kernel_var
        self.wz = (torch.randn(p, q, generator=g) * wz_scale).double().to(device)
        self.bias = torch.zeros(p, dtype=torch.float64, device=device)
        self.lengthscale = lengthscale
        self.device = device

    def kernel(self, t):
        return rbf_kernel(t, self.lengthscale, self.kernel_var).to(self.device)

    def sample_groups(self, t, B, seed=0):
        """B independent groups of K obs. Returns y (B,K,p) on self.device."""
        torch.manual_seed(seed)
        t = torch.as_tensor(t, dtype=torch.float64)
        K = len(t)
        L = torch.linalg.cholesky(self.kernel(t))
        eps = torch.randn(B, K, self.q, dtype=torch.float64, device=self.device)
        z = torch.einsum("ab,rbk->rak", L, eps)
        eta = (torch.einsum("rak,pk->rap", z, self.wz) + self.bias).clamp(max=10.0)
        return torch.poisson(torch.exp(eta))


def _per_obs_map(Y, W, b, q, s2=1.0):
    A = s2 * torch.eye(q, dtype=W.dtype, device=W.device) + W.T @ W
    rhs = (torch.log1p(Y) - b) @ W
    return torch.linalg.solve(A, rhs.transpose(-1, -2)).transpose(-1, -2)


def fit_gp_zqe(Yd, t, q, *, subset_K=None, kernel_var=1.0, lengthscale_init=1.0,
               steps=800, batch=256, lr=0.02, seed=0, wz_scale=0.7, true_W=None,
               device=None):
    """Composite ZQE for a grouped GP-GLLVM. ``Yd`` is (B, G, p): B groups each with
    G *fully GP-correlated* observations on grid ``t`` (G,). Generated once.

    During fitting we only ever touch a **random K-subset of each group's G points**
    (``subset_K``, default min(G,15)) — valid by the GP marginal theorem (any subset
    is N(0, K(t_subset))). So every step inverts only a K×K block, never G×G; ``K`` is
    the tunable cost/efficiency knob. Per-step cost is O(batch·K·p + K³), independent
    of both B and G. Returns W, b, lengthscale, history, time_sec."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    Yd = torch.as_tensor(Yd, dtype=torch.float64, device=device)
    t = torch.as_tensor(t, dtype=torch.float64, device=device)
    B, G, p = Yd.shape
    K = min(subset_K or 15, G)
    torch.manual_seed(seed)
    W = torch.nn.Parameter((torch.randn(p, q, device=device) * wz_scale).double())
    b = torch.nn.Parameter(torch.zeros(p, dtype=torch.float64, device=device))
    log_ell = torch.nn.Parameter(torch.tensor(float(np.log(lengthscale_init)), device=device))
    opt = torch.optim.Adam([W, b, log_ell], lr=lr)

    def Lc(ell, ts):                                   # K×K block on the SELECTED points
        return torch.linalg.cholesky(rbf_kernel(ts, ell, kernel_var).to(device))

    hist = {"ell": [], "procrustes": []}
    if true_W is not None:
        from gllvm.autofit import procrustes_error
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for it in range(steps):
        ell = log_ell.exp()
        gidx = torch.randint(0, B, (min(batch, B),), device=device)          # minibatch of groups
        tidx = torch.randperm(G, device=device)[:K]                          # random K-subset of the G points
        ts = t[tidx]
        Yb = Yd[gidx][:, tidx, :]                                            # (batch, K, p)
        with torch.no_grad():
            Ld = Lc(ell, ts)
            eps_d = torch.linalg.solve_triangular(Ld, _per_obs_map(Yb, W, b, q), upper=False)
            eps = torch.randn(Yb.shape[0], K, q, dtype=torch.float64, device=device)
            zq = torch.einsum("ab,rbk->rak", Ld, eps)                        # fantasy: only K, never G
            yq = torch.poisson(torch.exp((torch.einsum("rak,pk->rap", zq, W) + b).clamp(max=10)))
            eps_q = torch.linalg.solve_triangular(Ld, _per_obs_map(yq, W, b, q), upper=False)
        Lg = Lc(ell, ts)
        ed = torch.einsum("rak,pk->rap", torch.einsum("ab,rbk->rak", Lg, eps_d), W) + b
        eq = torch.einsum("rak,pk->rap", torch.einsum("ab,rbk->rak", Lg, eps_q), W) + b
        loss = -((torch.log1p(Yb) * ed).sum(-1).mean() - (torch.log1p(yq) * eq).sum(-1).mean())
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([W, b, log_ell], 5.0); opt.step()
        hist["ell"].append(ell.item())
        if true_W is not None:
            hist["procrustes"].append(procrustes_error(true_W, W.detach().cpu()))
    if device == "cuda":
        torch.cuda.synchronize()
    return {"W": W.detach().cpu(), "b": b.detach().cpu(),
            "lengthscale": float(log_ell.exp()), "history": hist,
            "time_sec": time.time() - t0, "subset_K": K}
