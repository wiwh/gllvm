"""
gp_fit.py — composite ZQE estimator for the GP-GLLVM (playground; no /src).

Recovers (W, b, lengthscale) from K-blocks only — never the full n×n. Design
(exactly mirroring the GLLVM ZQE recipe):

* The latent is the **white** ε; the kernel lives in the **decoder**:
  ``ε → z = L(ℓ) ε → η = W z + b``. So ℓ is in η and gets gradients by backprop.
* The **encoder is pure no-grad imputation**: per-obs Gaussian-MAP → Ẑ, then whiten
  ``ε̂ = L(ℓ)⁻¹ Ẑ``. Because ε̂ is detached, the ``L(ℓ)`` reused in the decoder
  (``η = W·L(ℓ)·ε̂ + b``) does **not** cancel → ℓ is identified.
* Plain centered ZQE loss ``-(m1 - m2)``; fantasies are K-blocks drawn from the
  current θ (start white, correlate by L(ℓ), Poisson) — super fast.
"""

from __future__ import annotations
import numpy as np
import torch
from gp_gllvm import rbf_kernel


def _per_obs_map(Y, W, b, q, s2=1.0):
    A = s2 * torch.eye(q, dtype=W.dtype) + W.T @ W
    rhs = (torch.log1p(Y) - b) @ W  # (...,K,q)
    return torch.linalg.solve(A, rhs.transpose(-1, -2)).transpose(-1, -2)


def fit_gp_zqe(
    data_blocks,
    t_block,
    q,
    *,
    kernel_var=1.0,
    lengthscale_init=1.0,
    steps=2000,
    lr=0.02,
    sigma2=1.0,
    seed=0,
    wz_scale=0.7,
    true_W=None,
    verbose=False,
):
    """Fit a GP-GLLVM by composite ZQE on ``data_blocks`` (B,K,p) at inputs ``t_block`` (K,).

    Returns a dict: ``W, b, lengthscale, history`` (history has per-step
    ``ell``, ``loss``, and ``procrustes`` if ``true_W`` is given)."""
    data_blocks = torch.as_tensor(data_blocks, dtype=torch.float64)
    t_block = torch.as_tensor(t_block, dtype=torch.float64)
    B, K, p = data_blocks.shape
    torch.manual_seed(seed)

    W = torch.nn.Parameter(torch.randn(p, q, dtype=torch.float64) * wz_scale)
    b = torch.nn.Parameter(torch.zeros(p, dtype=torch.float64))
    log_ell = torch.nn.Parameter(torch.tensor(float(np.log(lengthscale_init))))
    opt = torch.optim.Adam([W, b, log_ell], lr=lr)

    def Lchol(ell):
        return torch.linalg.cholesky(rbf_kernel(t_block, ell, kernel_var))

    hist = {"ell": [], "loss": [], "procrustes": []}
    if true_W is not None:
        from gllvm.autofit import procrustes_error

    for t in range(steps):
        ell = log_ell.exp()
        with torch.no_grad():  # encoder = imputation
            Ld = Lchol(ell)
            eps_d = torch.linalg.solve_triangular(
                Ld, _per_obs_map(data_blocks, W, b, q), upper=False
            )
            eps = torch.randn(
                B, K, q, dtype=torch.float64
            )  # fantasy: white → correlate → Poisson
            zq = torch.einsum("ab,rbk->rak", Ld, eps)
            yq = torch.poisson(
                torch.exp((torch.einsum("rak,pk->rap", zq, W) + b).clamp(max=10))
            )
            eps_q = torch.linalg.solve_triangular(
                Ld, _per_obs_map(yq, W, b, q), upper=False
            )
        Lg = Lchol(ell)  # decoder: WITH grad
        eta_d = (
            torch.einsum("rak,pk->rap", torch.einsum("ab,rbk->rak", Lg, eps_d), W) + b
        )
        eta_q = (
            torch.einsum("rak,pk->rap", torch.einsum("ab,rbk->rak", Lg, eps_q), W) + b
        )
        m1 = (torch.log1p(data_blocks) * eta_d).sum(-1).mean()
        m2 = (torch.log1p(yq) * eta_q).sum(-1).mean()
        loss = -(m1 - m2)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([W, b, log_ell], 5.0)
        for g in opt.param_groups:
            g["lr"] *= 0.995
        opt.step()
        hist["ell"].append(ell.item())
        hist["loss"].append(loss.item())
        if true_W is not None:
            hist["procrustes"].append(procrustes_error(true_W, W))
        if verbose and (t + 1) % 500 == 0:
            msg = f"  step {t+1:4d}  ell={ell.item():.3f}  loss={loss.item():+.4f}"
            if true_W is not None:
                msg += f"  procrustes={hist['procrustes'][-1]:.3f}"
            print(msg, flush=True)

    return {
        "W": W.detach(),
        "b": b.detach(),
        "lengthscale": float(log_ell.exp()),
        "history": hist,
    }
