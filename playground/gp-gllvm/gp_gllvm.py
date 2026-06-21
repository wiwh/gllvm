"""
gp_gllvm.py — standalone GP-GLLVM prototype (playground; does NOT touch /src).

Model (Poisson log-link demo)
-----------------------------
Observations are indexed by an input ``t_i`` (time / location / pseudotime).
Each latent factor is a Gaussian process over the inputs instead of iid:

    eps[:,k] ~ N(0, I_n)                  (independent "white" realization)
    z[:,k]   = L eps[:,k],  L = chol(K(t))   →  z[:,k] ~ GP(0, K(t))   (correlated)
    y_i      ~ Poisson( exp(W z_i + b) )

So marginally each ``z_i ~ N(0, I)`` (standard GLLVM), but the latents are
**correlated across observations**. Key GP fact used everywhere here: the marginal
of the GP over *any* subset of K points is exactly ``N(0, K_block)`` — so a block of
K observations is itself a valid GP draw. That is what lets the estimator work with
tiny K×K blocks (K≈10–20) and never invert the full n×n.

Encoder (one extra step vs the ordinary Gaussian-MAP)
----------------------------------------------------
For a block of K observations ``(y_block, t_block)``:
  1. per-observation Gaussian-MAP  →  ``Zhat`` (K×q), *correlated* (it estimates the
     correlated z's);
  2. **de-correlate / whiten** with the block kernel:
     ``eps_hat = L_block^{-1} Zhat``  (L_block = chol(K(t_block)))  →  the independent
     realization ε (the generator's starting point).

Generator vs fantasies
----------------------
The *true* data is generated with full correlation over n≈200–500 (invertible well).
The estimator's *fantasy* samples are always small K-blocks (K≈10–20) — super fast.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


def rbf_kernel(t, lengthscale, variance=1.0, jitter=1e-3):
    """RBF/squared-exponential kernel over 1-D inputs ``t`` (n,) → (n, n)."""
    t = torch.as_tensor(t, dtype=torch.float64).reshape(-1)
    d2 = (t[:, None] - t[None, :]) ** 2
    K = variance * torch.exp(-0.5 * d2 / (lengthscale ** 2))
    return K + jitter * torch.eye(len(t), dtype=K.dtype, device=K.device)


class GPGLLVM(nn.Module):
    """Gaussian-process latent GLLVM (Poisson log-link)."""

    def __init__(self, latent_dim, output_dim, lengthscale=2.0, kernel_var=1.0,
                 wz_scale=0.7, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.q, self.p = latent_dim, output_dim
        self.wz = nn.Parameter(torch.randn(output_dim, latent_dim, generator=g) * wz_scale)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.log_ell = nn.Parameter(torch.tensor(float(np.log(lengthscale))))
        self.log_var = nn.Parameter(torch.tensor(float(np.log(kernel_var))))

    @property
    def lengthscale(self):
        return self.log_ell.exp().item()

    @property
    def kernel_var(self):
        return self.log_var.exp().item()

    def kernel(self, t):
        return rbf_kernel(t, self.lengthscale, self.kernel_var)

    # ---- generative ----
    def sample_batch(self, t, R=1, seed=None):
        """Draw ``R`` independent GP-GLLVM realizations over inputs ``t`` (n,).

        Returns ``y (R,n,p) float, z (R,n,q), eps (R,n,q)``. The Cholesky of the
        n×n kernel is computed once and reused across the R draws.
        """
        if seed is not None:
            torch.manual_seed(seed)
        t = torch.as_tensor(t, dtype=torch.float64).reshape(-1)
        n = len(t)
        L = torch.linalg.cholesky(self.kernel(t))                 # (n,n)
        eps = torch.randn(R, n, self.q, dtype=torch.float64)       # white
        z = torch.einsum("ab,rbk->rak", L, eps)                   # correlate per column
        W = self.wz.double(); b = self.bias.double()
        eta = (torch.einsum("rak,pk->rap", z, W) + b).clamp(max=10.0)
        y = torch.poisson(torch.exp(eta))
        return y, z, eps

    def sample(self, t, seed=None):
        y, z, eps = self.sample_batch(t, R=1, seed=seed)
        return y[0], z[0], eps[0]


class GPMapEncoder:
    """Per-observation Gaussian-MAP (on log1p) → correlated Ẑ, then whiten by the
    block kernel → independent ε̂. Holds a live reference to the model (uses current
    W, b, kernel params)."""

    def __init__(self, gp: GPGLLVM, sigma2=1.0):
        self.gp = gp
        self.sigma2 = sigma2

    def map_zhat(self, y_block):
        """Per-obs Gaussian-MAP on log1p: (K,p) → (K,q) correlated latent estimate."""
        W = self.gp.wz.double(); b = self.gp.bias.double()
        t_y = torch.log1p(torch.as_tensor(y_block, dtype=torch.float64))
        A = self.sigma2 * torch.eye(self.gp.q, dtype=torch.float64) + W.T @ W
        rhs = (t_y - b) @ W                                       # (K,q)
        return torch.linalg.solve(A, rhs.T).T                     # (K,q)

    def whiten(self, Zhat, t_block):
        """De-correlate the block: eps_hat = L_block^{-1} Zhat   (L_block = chol K_block)."""
        Lb = torch.linalg.cholesky(self.gp.kernel(t_block))       # (K,K)
        return torch.linalg.solve_triangular(Lb, Zhat, upper=False)

    def encode_block(self, y_block, t_block):
        """Return (Ẑ correlated, ε̂ whitened) for a block of K observations."""
        Zhat = self.map_zhat(y_block)
        return Zhat, self.whiten(Zhat, t_block)
