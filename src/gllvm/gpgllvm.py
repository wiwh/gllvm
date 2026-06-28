"""
GP-GLLVM: a GLLVM with a Gaussian-process latent prior over coordinates.

The decoder is exactly a :class:`~gllvm.gllvm_module.GLLVM` (each response a GLM
family, ``y | z`` unchanged), so :class:`GPGLLVM` *inherits* ``forward`` /
``log_prob`` / ``zq_log`` / ``mean`` and the whole family machinery.  What changes
is the **latent prior**: instead of ``z ~ N(0, I)`` i.i.d., each latent factor is an
independent Gaussian process over input coordinates ``t`` (``B = I``):

    z[:, k] ~ GP(0, K_k),   k = 1..q     (factors independent; per-factor kernel)

so the joint latent covariance ``Σ(ℓ)`` is **block-diagonal** across factors.

This module provides three pieces, mirroring the GLLVM stack
(model / encoder / fitter):

* :class:`GPGLLVM`      — the model (decoder + GP prior, ``RBFKernel`` by default).
* :class:`GPMapEncoder` — parameter-free joint block-MAP imputer (Gaussian log1p proxy).
* :class:`GPZQEFitter`  — likelihood-free ZQE fitter (K-subset patches; centered
  loadings + fantasy-centered second-moment length-scale fit).

Identification note: the loadings ``W`` are kept **full** (no lower-triangular
constraint).  With distinct ``ℓ_k`` the kernels already break the rotation gauge,
and a lower-triangular constraint would *exclude the truth* and bias both ``W`` and
``ℓ`` — see ``paper/CLAUDE.md``.  Scaling note: with ``B = I`` generating/whitening
are exactly per-factor (linear in ``q``); the encoder solve couples factors via
``WᵀW`` (dense here — a CG variant for very large ``q`` is left for later).
"""

import time
from typing import Optional

import torch
import torch.nn as nn

from gllvm.gllvm_module import GLLVM
from gllvm.kernels import RBFKernel


# ----------------------------------------------------------------------------
# vec / unvec (factor-major) and block-diagonal assembly
# ----------------------------------------------------------------------------
def _vec(Z: torch.Tensor) -> torch.Tensor:
    """(*batch, K, q) -> (*batch, qK), factor-major: [factor0 over K; factor1; ...]."""
    return Z.transpose(-1, -2).reshape(*Z.shape[:-2], -1)


def _unvec(u: torch.Tensor, q: int, K: int) -> torch.Tensor:
    """(*batch, qK) -> (*batch, K, q), inverse of :func:`_vec`."""
    return u.reshape(*u.shape[:-1], q, K).transpose(-1, -2)


def _block_diag(blocks: torch.Tensor) -> torch.Tensor:
    """(*batch, q, K, K) -> (*batch, qK, qK) block-diagonal (factor-major)."""
    *b, q, K, _ = blocks.shape
    out = blocks.new_zeros(*b, q * K, q * K)
    for k in range(q):
        out[..., k * K:(k + 1) * K, k * K:(k + 1) * K] = blocks[..., k, :, :]
    return out


# ============================================================================
# Model
# ============================================================================
class GPGLLVM(GLLVM):
    """Generalised Latent Variable Model with a Gaussian-process latent prior.

    Same decoder as :class:`~gllvm.gllvm_module.GLLVM` (add families with
    :meth:`add_glm`); the latent ``z`` is a GP realisation over coordinates rather
    than i.i.d. ``N(0, I)``.  Defaults to ``float64`` (GP Cholesky stability).

    Parameters
    ----------
    latent_dim, output_dim : int
        Latent dimension ``q`` and number of responses ``p``.
    input_dim : int
        Coordinate dimension ``d`` (1 = time, 2 = space, ...).
    kernel : Kernel, optional
        Latent covariance kernel; defaults to :class:`~gllvm.kernels.RBFKernel`
        with ``lengthscale`` / ``jitter``.
    lengthscale, jitter : passed to the default ``RBFKernel`` if ``kernel`` is None.
    feature_dim, bias : as in :class:`GLLVM`.
    dtype : torch dtype (default ``float64``).
    """

    def __init__(self, latent_dim: int, output_dim: int, *, input_dim: int = 1,
                 kernel=None, lengthscale=1.0, jitter: float = 1e-4,
                 feature_dim: int = 0, bias: bool = True, dtype=torch.float64):
        super().__init__(latent_dim, output_dim, feature_dim=feature_dim, bias=bias)
        self.input_dim = input_dim
        self.kernel = kernel if kernel is not None else RBFKernel(latent_dim, lengthscale, jitter)
        self.to(dtype)

    @property
    def lengthscale(self) -> torch.Tensor:
        """Per-factor length-scales ``(q,)`` of the latent kernel."""
        return self.kernel.lengthscale

    # ---- GP latent prior --------------------------------------------------
    def sample_z(self, coords: torch.Tensor) -> torch.Tensor:
        """Draw ``z ~ GP(0, Σ(ℓ))`` at ``coords`` ``(*batch, K, d)`` → ``(*batch, K, q)``.

        Exact per-factor (block-diagonal Σ): ``z[..., k] = L_k ε_k``, ``ε ~ N(0, I)``.
        (Overrides ``GLLVM.sample_z(num_samples)`` — the GP prior needs coordinates.)
        """
        coords = coords.to(self.wz.dtype)
        L = self.kernel.cholesky(coords)                              # (*b, q, K, K)
        eps = torch.randn(*coords.shape[:-1], self.q, dtype=self.wz.dtype, device=self.wz.device)
        z = torch.einsum("...qij,...qj->...qi", L, eps.transpose(-1, -2))
        return z.transpose(-1, -2)                                    # (*b, K, q)

    def whiten(self, z: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Whiten ``ε = L_Σ⁻¹ z`` per factor (inverse of :meth:`sample_z`'s correlate)."""
        L = self.kernel.cholesky(coords.to(z.dtype))
        zf = z.transpose(-1, -2).unsqueeze(-1)                        # (*b, q, K, 1)
        eps = torch.linalg.solve_triangular(L, zf, upper=False).squeeze(-1)
        return eps.transpose(-1, -2)

    def cov(self, coords: torch.Tensor) -> torch.Tensor:
        """Block-diagonal latent covariance ``Σ(coords)`` ``(*batch, qK, qK)``."""
        return _block_diag(self.kernel.blocks(coords.to(self.wz.dtype)))

    # ---- decoder (offset-aware) ------------------------------------------
    def forward(self, z, x=None, offset=None):
        """Linear predictor ``η = z Wᵀ (+ x Wxᵀ) + b (+ offset)``.

        ``offset`` (a known per-observation additive term in ``η``, e.g.
        ``log`` library size) broadcasts over responses.
        """
        linpar = super().forward(z, x)
        if offset is not None:
            linpar = linpar + offset.unsqueeze(-1)
        return linpar

    def sample(self, coords=None, *, z=None, x=None, offset=None):
        """Sample ``y`` at ``coords`` (draws ``z`` from the GP prior if not given).

        Works for any leading batch shape ``(*batch, K, ...)``.
        """
        self._check_assignments()
        if z is None:
            if coords is None:
                raise ValueError("GPGLLVM.sample needs either coords or z.")
            z = self.sample_z(coords)
        linpar = self.forward(z, x, offset=offset)
        y = torch.empty_like(linpar)
        for glm in self.families:
            idx = glm.idx
            y[..., idx] = glm(linpar=linpar[..., idx], scale=self.scale[idx]).sample()
        return y

    def __repr__(self):
        fams = ", ".join(f.name or f.GLM.__name__ for f in self.families) or "(none)"
        return (f"GPGLLVM(q={self.q}, p={self.p}, input_dim={self.input_dim},\n"
                f"  kernel={self.kernel},\n"
                f"  families=[{fams}])")


# ============================================================================
# Encoder
# ============================================================================
class GPMapEncoder(nn.Module):
    r"""Parameter-free joint block-MAP encoder for the GP-GLLVM.

    Under the Gaussian ``log1p`` proxy
    ``log1p(y) | z ~ N(Wz + b + offset, σ²I)``, ``z ~ N(0, Σ(ℓ))``, the whole
    ``K``-block is imputed *jointly*:

    .. math::
        \hat z = \big(\Sigma^{-1} + (W^\top W/\sigma^2)\otimes I_K\big)^{-1}
                 \operatorname{vec}\!\big((\log1p(y) - b - \text{offset})\,W/\sigma^2\big),

    with ``Σ⁻¹`` block-diagonal (``B = I``), built per factor.  Holds a live
    reference to the model (always uses the current ``W, b, ℓ``).  A deterministic
    δ-mass surrogate like the GLLVM MAP encoders; the fitter calls it under
    ``no_grad`` (score-function identity → the encoder is a detached surrogate).
    """

    def __init__(self, gpgllvm: GPGLLVM, sigma2: float = 1.0):
        super().__init__()
        self.gp = gpgllvm
        self.sigma2 = sigma2

    def forward(self, y, coords, offset=None):
        g = self.gp
        W = g.wz * g.wz_mask
        b = g.bias if g.bias is not None else torch.zeros(g.p, dtype=W.dtype, device=W.device)
        q, K, s2 = g.q, y.shape[-2], self.sigma2

        t = torch.log1p(y.to(W.dtype)) - b
        if offset is not None:
            t = t - offset.unsqueeze(-1)

        Kq = g.kernel.blocks(coords.to(W.dtype))                      # (*b, q, K, K)
        Lk = torch.linalg.cholesky(Kq)
        eye = torch.eye(K, dtype=W.dtype, device=W.device)
        Kinv = torch.cholesky_solve(eye.expand_as(Kq), Lk)           # per-block Σ⁻¹
        A = _block_diag(Kinv) + torch.kron(W.T @ W / s2, eye)        # (*b, qK, qK)
        rhs = _vec(t @ W / s2)                                        # (*b, qK)
        z = torch.cholesky_solve(rhs.unsqueeze(-1), torch.linalg.cholesky(A)).squeeze(-1)
        return _unvec(z, q, K)

    def sample(self, y, coords, offset=None):
        """Drop-in surrogate: returns ``(ẑ, ẑ, -inf)`` (deterministic δ-mass)."""
        z = self.forward(y, coords, offset)
        return z, z, torch.full_like(z, float("-inf"))


# ============================================================================
# Fitter
# ============================================================================
class GPZQEFitter:
    r"""Likelihood-free ZQE fitter for the :class:`GPGLLVM` (``B = I``, per-factor ℓ).

    Estimating equations on random **K-subsets of each group** (GP marginal
    theorem: any subset of a GP is an exact GP, so every operation is ``K x K``,
    never ``G x G`` — cost is independent of group size):

    * **loadings** ``(W, b)``: centered ``-(m1 - m2)`` with
      ``m = E[ T(y)·η(ẑ) ]`` (real data minus model-fantasy — the centering that
      fixes the population root regardless of the encoder).
    * **length-scales** ``ℓ``: a fantasy-centered **second-moment** match of the
      imputed latent, ``Σ(ℓ) ↔ E[ẑ ẑᵀ]`` per patch — the cross-observation moment
      that identifies the kernel (a per-observation ``T(y)·η`` cannot see ``ℓ``).

    The encoder is a no-grad :class:`GPMapEncoder` (detached; score-function
    identity).  ``W`` is kept full (distinct ``ℓ_k`` identify it).

    Parameters
    ----------
    model : GPGLLVM
    encoder : nn.Module, optional
        Parameter-free encoder with ``.forward(y, coords, offset)``; defaults to
        :class:`GPMapEncoder`.
    K : int
        Sub-sample size per group per step.
    steps, lr, batch, warmup : optimiser schedule (Adam).  ``warmup`` steps fit the
        loadings only (length-scales held), then both.
    fit_lengthscale : bool
        If ``False``, the kernel length-scales are frozen (e.g. a fixed-ℓ scan).
    cov_batch : int
        Number of patches used for the (memory-heavier) second-moment term.
    grad_clip, device, seed, verbose : misc.
    """

    def __init__(self, model: GPGLLVM, *, encoder=None, sigma2: float = 1.0, K: int = 64,
                 steps: int = 2000, lr: float = 0.03, batch: int = 128, warmup: int = 200,
                 fit_lengthscale: bool = True, cov_batch: int = 32, grad_clip: float = 5.0,
                 device: Optional[str] = None, seed: int = 0, verbose: bool = False):
        self.model = model
        self.encoder = encoder if encoder is not None else GPMapEncoder(model, sigma2=sigma2)
        self.K, self.steps, self.lr = K, steps, lr
        self.batch, self.warmup = batch, warmup
        self.fit_lengthscale, self.cov_batch = fit_lengthscale, cov_batch
        self.grad_clip, self.seed, self.verbose = grad_clip, seed, verbose
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # -- group/patch bookkeeping -------------------------------------------
    def _setup(self, y, coords, groups, offset):
        dev, dt = self.device, self.model.wz.dtype
        self.model.to(dev)
        self.y = torch.as_tensor(y, dtype=dt, device=dev)
        c = torch.as_tensor(coords, dtype=dt, device=dev)
        self.coords = c[:, None] if c.ndim == 1 else c
        self.offset = None if offset is None else torch.as_tensor(offset, dtype=dt, device=dev)
        n = self.y.shape[0]
        if groups is None:
            self._groups = [torch.arange(n, device=dev)]
        else:                                            # sort-based group-by: O(n log n), any #groups
            g = torch.as_tensor(groups, device=dev)
            order = torch.argsort(g, stable=True)
            _, counts = torch.unique_consecutive(g[order], return_counts=True)
            self._groups = list(torch.split(order, counts.tolist()))
        gmin = min(len(g) for g in self._groups)
        if self.K > gmin:
            self.K = gmin            # can't draw more points than the smallest group has

    def _patch(self):
        gi = torch.randint(0, len(self._groups), (self.batch,))
        rows = [self._groups[g][torch.randperm(len(self._groups[g]), device=self.device)[:self.K]]
                for g in gi.tolist()]
        return torch.stack(rows)     # (batch, K)

    # -- fit ---------------------------------------------------------------
    def fit(self, y, coords, groups=None, offset=None):
        """Fit on responses ``y`` ``(n, p)`` at ``coords`` ``(n, d)``.

        ``groups`` ``(n,)`` (optional) marks independent GP groups (default: one);
        ``offset`` ``(n,)`` (optional) is a known additive term in ``η``.
        Returns ``self`` (``self.model`` fitted in place; ``self.history``,
        ``self.lengthscales_``).
        """
        torch.manual_seed(self.seed)
        self._setup(y, coords, groups, offset)
        g, p, q = self.model, self.model.p, self.model.q

        params = [g.wz, g.kernel.log_lengthscale] + ([g.bias] if g.bias is not None else [])
        opt = torch.optim.Adam(params, lr=self.lr)
        hist = {k: [] for k in ("loss", "load", "cov", "ell")}
        t0 = time.time()

        for step in range(self.steps):
            idx = self._patch()
            yb, cb = self.y[idx], self.coords[idx]
            ob = self.offset[idx] if self.offset is not None else None

            with torch.no_grad():
                zq = g.sample_z(cb)
                yq = g.sample(z=zq, offset=ob)
                zd = self.encoder.forward(yb, cb, ob)
                zh = self.encoder.forward(yq, cb, ob)

            # loadings: centered -(m1 - m2), per observation (flatten the patches)
            ob_f = None if ob is None else ob.reshape(-1)
            m1 = g.zq_log(yb.reshape(-1, p),
                          linpar=g.forward(zd.reshape(-1, q), offset=ob_f)).sum(-1).mean()
            m2 = g.zq_log(yq.reshape(-1, p),
                          linpar=g.forward(zh.reshape(-1, q), offset=ob_f)).sum(-1).mean()
            loss = -(m1 - m2)
            load_val = loss.item()

            cov_val = 0.0
            if self.fit_lengthscale:
                cb_, zd_, zh_ = cb[:self.cov_batch], zd[:self.cov_batch], zh[:self.cov_batch]
                Sig = g.cov(cb_)                                   # (cb, qK, qK), has grad in ℓ
                with torch.no_grad():
                    vd, vh = _vec(zd_), _vec(zh_)                  # (cb, qK)
                    disc = (vd[..., :, None] * vd[..., None, :]
                            - vh[..., :, None] * vh[..., None, :])  # per-patch fantasy-centered outer
                    target = Sig.detach() + disc
                cov_loss = ((Sig - target) ** 2).mean()
                loss = loss + cov_loss
                cov_val = cov_loss.item()

            opt.zero_grad()
            loss.backward()
            if step < self.warmup and self.fit_lengthscale:
                g.kernel.log_lengthscale.grad = None
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(params, self.grad_clip)
            opt.step()

            hist["loss"].append(loss.item()); hist["load"].append(load_val)
            hist["cov"].append(cov_val); hist["ell"].append(g.lengthscale.detach().cpu().tolist())
            if self.verbose and step % max(1, self.steps // 10) == 0:
                ell = [round(x, 3) for x in g.lengthscale.detach().cpu().tolist()]
                print(f"[{step:>5}/{self.steps}] load={load_val:+.4f} cov={cov_val:.2e} ell={ell}")

        self.history = hist
        self.lengthscales_ = g.lengthscale.detach()
        self.fit_time_ = time.time() - t0
        if self.verbose:
            print(f"done {self.fit_time_:.0f}s  ell={[round(x,3) for x in self.lengthscales_.cpu().tolist()]}")
        return self
