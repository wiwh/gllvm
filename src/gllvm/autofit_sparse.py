"""
Sparse / overcomplete ZQE — *sparsity discovery* for the loadings (elastic-net L1 + L2).

Separate from ``autofit.py`` (the trusted dense estimator), and aimed at a *different job*:
given an overcomplete fit (``q_fit > q_true``, possibly q > p), **discover which loadings are
zero** — i.e. recover a support mask and the effective factor count.  This is a one-pass
discovery procedure, NOT a precision estimator.  The precise fit is a separate downstream
step: take the mask this returns, freeze it (``g.set_wz_mask(mask)``), and run the normal
``ZQEAutoFitter`` on the fixed support.  So there is deliberately **no restart/convergence
machinery here** — one sparsification pass, read off the mask.

Design points:

* **No lower-tri identification.**  A sparse pattern may only exist in *one* rotation; the
  lower-triangular gauge constraint locks the rotation and can hide that pattern.  We keep W
  fully free and let L1 pick the sparse rotation.  (Never call ``set_wz_mask`` with a
  triangular mask in this fitter.)

* **Elastic net, not just L1.**  A small L2 ridge keeps each step well-conditioned so the
  path stays computable as λ₁ → 0 (the dense limit) — its only job is conditioning, so it is
  small (a multiplicative shrink ``W /= 1 + lr·l2``; large l2 over-shrinks and destabilises).

* **The zero-column absorbing state.**  A column the prox fully zeroes is *stuck*: the
  parameter-free encoder returns ``z_k = 0`` for a zero column (prior wins), so
  ``∂loss/∂W_·k = (…)·z_k = 0`` — no gradient, never revives.  We re-seed dead columns with
  small noise during the exploration phase so they get re-tested.  Only whole columns need
  this — a zeroed *entry* in a live column already gets a non-zero gradient.

* **Support from the averaged gradient.**  Prox-SGD + iterate-averaging smears per-step zeros,
  and the per-step iterate count is lr-dependent (a zeroed entry revives at ≈lr·g, so zeros get
  sticky as lr decays).  We read the *support* off the tail-averaged gradient instead — the
  proximal stationarity condition ``|ḡ_j| ≤ λ₁  ⇒  W_j = 0`` on the de-noised mean gradient,
  which is lr-free — and keep the averaged magnitudes on the survivors.  ``self.mask_`` is the
  result.  (This is discovery, not inference: we want a robust support, not a calibrated SE.)

* **λ-annealing (optional).**  Start dense-penalised (λ₁ high → most columns dead) and anneal
  λ₁ down across the exploration phase, activating columns as the penalty drops.

Entry point: ``SparseZQEFitter(gllvm, l1=…, l2=…, …).fit(y)`` → ``.mask_``, ``.model``.
"""

import copy
import math
from typing import Callable, Optional, Sequence

import numpy as np
import torch

from .autofit import procrustes_error, orthogonal_align  # noqa: F401  (re-exported)
from .encoder import MapEncoderGaussianLog1p


def _soft_threshold(W: torch.Tensor, t: float) -> torch.Tensor:
    """Element-wise soft-threshold prox of the L1 norm: sign(W)·max(|W|−t, 0)."""
    return torch.sign(W) * (W.abs() - t).clamp_min(0.0)


class SparseZQEFitter:
    """Elastic-net ZQE for sparse / overcomplete loadings.

    Parameters
    ----------
    gllvm : GLLVM
        Model to fit in place (deep-copied internally; ``.model`` holds the result).
    encoder_factory : callable, optional
        ``g -> encoder`` exposing ``.sample(y) -> (z, mu, logvar)``.  Default
        ``MapEncoderGaussianLog1p``; pass ``MapEncoderPoissonNewton`` for the proper
        Poisson MAP (better z for downstream use).
    l1 : float
        L1 penalty (the selection knob).  Per-step prox threshold is ``l1·lr``; the final
        support is ``|ḡ_j| > l1`` on the tail-averaged gradient.
    l2 : float
        L2 ridge (elastic-net) — keeps the step well-conditioned and the λ₁→0 limit
        computable.  Applied as a multiplicative shrink ``W /= (1 + lr·l2)`` each step.
        Keep it *small* (conditioning only); large l2 over-shrinks W and destabilises.
    anneal_l1_from : float, optional
        If set, λ₁ is annealed log-linearly from this value down to ``l1`` across the
        exploration phase (homotopy from sparse → target).  ``None`` → constant ``l1``.
    start_dead : bool
        Zero the loadings at the start of refinement (then probe/anneal back to life).
    steps : int
        Total refine steps.  The first ``polyak_frac`` are *exploration* (annealing +
        revival, LR decaying); the rest are *averaging* (λ₁ fixed, Polyak tail).
    refine_lr, refine_lr_power : float
        SGD LR and within-chain decay exponent ``lr_k = refine_lr/(1+k)**power``.
    warmup_lr, warmup_epochs : float, int
        Dense unpenalised Adam warm-up (fixes bias + rough scale so the encoder is
        meaningful before sparsification).  ``warmup_epochs=0`` skips it.
    revive_every, revive_jitter : int, float
        During exploration, re-seed fully-dead columns with ``N(0, revive_jitter²)`` every
        ``revive_every`` steps.  ``revive_jitter=0`` disables revival.
    """

    def __init__(
        self,
        gllvm,
        *,
        encoder_factory: Optional[Callable] = None,
        l1: float = 0.05,
        l2: float = 0.01,
        anneal_l1_from: Optional[float] = None,
        start_dead: bool = False,
        steps: int = 800,
        refine_lr: float = 0.3,
        refine_lr_power: float = 0.5,
        warmup_lr: float = 0.1,
        warmup_l2: float = 1e-3,
        warmup_refine_steps: int = 100,
        warm_model=None,  # skip the internal warm-start, start from this GLLVM (λ-path reuse)
        batch_size: Optional[int] = None,
        sim_factor: float = 1.0,
        polyak_frac: float = 0.5,
        revive_every: int = 100,
        revive_jitter: float = 0.1,
        max_consecutive_fails: int = 10,
        grad_clip: float = 5.0,
        device: str = "cpu",
        seed: Optional[int] = None,
        verbose: bool = True,
    ):
        self.gllvm = gllvm.to(device)
        self.device = device
        self.encoder_factory = encoder_factory or (lambda g: MapEncoderGaussianLog1p(g))
        self.l1 = l1
        self.l2 = l2
        self.anneal_l1_from = anneal_l1_from
        self.start_dead = start_dead
        self.steps = steps
        self.refine_lr = refine_lr
        self.refine_lr_power = refine_lr_power
        self.warmup_lr = warmup_lr
        self.warmup_l2 = warmup_l2
        self.warmup_refine_steps = warmup_refine_steps
        self.warm_model = warm_model
        self.batch_size = batch_size
        self.sim_factor = sim_factor
        self.polyak_frac = polyak_frac
        self.revive_every = revive_every
        self.revive_jitter = revive_jitter
        self.max_consecutive_fails = max_consecutive_fails
        self.grad_clip = grad_clip
        self.seed = seed
        self.verbose = verbose

        # ---- results (filled by fit) ----
        self.model = None        # discovered loadings: avg magnitudes on the KKT support
        self.mask_ = None        # (p, q) bool — True = surviving (nonzero) loading; THE OUTPUT
        self.gbar_ = None        # (p, q) tail-averaged gradient (the estimating function)
        self.history = {"loss": [], "l1": []}

    # ------------------------------------------------------------------
    # ZQE gradient (same estimating function as ZQEAutoFitter, encoder detached)
    # ------------------------------------------------------------------
    def _zqe_loss(self, g, enc, yb, n_sim):
        with torch.no_grad():
            yq = g.sample(z=g.sample_z(n_sim))
            z, _, _ = enc.sample(yb)        # detached — score-function identity
            z_q, _, _ = enc.sample(yq)
        zq1 = g.zq_log(yb, z=z)
        zq2 = g.zq_log(yq, z=z_q)
        return -(zq1.sum(-1).mean() - zq2.sum(-1).mean())

    def _revive(self, g):
        """Re-seed fully-dead columns so the encoder produces a non-zero z_k → testable."""
        if self.revive_jitter <= 0.0:
            return 0
        with torch.no_grad():
            dead = g.wz.norm(dim=0) < 1e-8
            n = int(dead.sum())
            if n:
                g.wz.data[:, dead] = self.revive_jitter * torch.randn(
                    g.p, n, device=g.wz.device, dtype=g.wz.dtype
                )
        return n

    # ------------------------------------------------------------------
    def _warm_start(self, y):
        """Dense, unpenalised warm start — delegated to the trusted ZQEAutoFitter (its
        annealed/stabilised warm-up + a short refine), so the encoder is meaningful before
        we start sparsifying.  Returns a fresh GLLVM with good (W, b)."""
        from .autofit import ZQEAutoFitter

        warm = ZQEAutoFitter(
            copy.deepcopy(self.gllvm),
            encoder_factory=self.encoder_factory,
            l2=self.warmup_l2,
            device=self.device,
            warmup_lr=self.warmup_lr,
            refine_lr=self.refine_lr,
            steps_per_round=self.warmup_refine_steps,
            max_rounds=1,
            store_wz_trace=False,
            seed=self.seed,
            verbose=self.verbose,
        ).fit(y)
        return copy.deepcopy(warm.model).to(self.device)

    # ------------------------------------------------------------------
    def fit(self, y):
        if self.seed is not None:
            torch.manual_seed(self.seed)
        y = y.to(self.device)
        n = len(y)
        bs = self.batch_size or n
        n_sim = max(1, round(self.sim_factor * bs))

        # 1) dense unpenalised warm start (bias + scale).  λ-independent → reusable across a
        # whole λ-path: pass warm_model to skip it.  self.warm_model_ exposes it for reuse.
        if self.warm_model is not None:
            g = copy.deepcopy(self.warm_model).to(self.device)
        else:
            g = self._warm_start(y)
        self.warm_model_ = copy.deepcopy(g)
        enc = self.encoder_factory(g)
        if self.start_dead:
            with torch.no_grad():
                g.wz.zero_()
            self._revive(g)  # tiny probe so the first gradients are informative

        # 2) single elastic-net sparsification pass (exploration → averaging)
        explore = int(self.polyak_frac * self.steps)
        l1_hi = self.anneal_l1_from if self.anneal_l1_from is not None else self.l1
        Wsum = torch.zeros_like(g.wz)
        bsum = None if g.bias is None else torch.zeros_like(g.bias)
        gsum = torch.zeros_like(g.wz)
        gbar_run = torch.zeros_like(g.wz)  # EMA of the raw gradient — the lr-free support
        navg = 0
        fail_streak = 0  # consecutive bad steps (encoder NaN etc.) → give up only if sustained

        for k in range(self.steps):
            lr = self.refine_lr / (1.0 + k) ** self.refine_lr_power
            # exploration: anneal λ₁ high→target (log scale); averaging: hold at target
            if k < explore and self.anneal_l1_from is not None:
                frac = k / max(1, explore - 1)
                l1_k = math.exp(
                    (1 - frac) * math.log(l1_hi + 1e-12) + frac * math.log(self.l1 + 1e-12)
                )
            else:
                l1_k = self.l1

            yb = y if bs >= n else y[torch.randperm(n, device=self.device)[:bs]]
            g.wz.grad = None
            if g.bias is not None:
                g.bias.grad = None
            # last-good snapshot: the encoder Newton can occasionally return a bad z for an
            # intermediate W, which makes the Poisson sampler raise on a NaN rate.  Restore
            # and stop the pass (the mask from accumulated averaging is still valid).
            wz0 = g.wz.detach().clone()
            b0 = None if g.bias is None else g.bias.detach().clone()
            try:
                loss = self._zqe_loss(g, enc, yb, n_sim)
                if not torch.isfinite(loss):
                    raise ValueError("non-finite loss")
                loss.backward()
            except (ValueError, RuntimeError) as e:
                # bad step (e.g. encoder Newton diverged at this W): restore last-good,
                # perturb dead columns, and CONTINUE — abort only if it keeps failing
                with torch.no_grad():
                    g.wz.copy_(wz0)
                    if g.bias is not None:
                        g.bias.copy_(b0)
                self._revive(g)
                fail_streak += 1
                if fail_streak >= self.max_consecutive_fails:
                    if self.verbose:
                        print(f"  step {k+1}: {e} — {fail_streak} consecutive fails, stopping")
                    break
                continue
            torch.nn.utils.clip_grad_norm_(g.parameters(), self.grad_clip)
            gW = g.wz.grad.detach().clone()

            with torch.no_grad():
                # elastic-net prox-SGD: grad step → L2 shrink → L1 soft-threshold
                g.wz -= lr * g.wz.grad
                if self.l2:
                    g.wz /= (1.0 + lr * self.l2)
                g.wz.copy_(_soft_threshold(g.wz, l1_k * lr))
                if g.bias is not None:
                    g.bias -= lr * g.bias.grad  # intercept: unpenalised
                if not torch.isfinite(g.wz).all():  # bad update → revert, perturb, continue
                    g.wz.copy_(wz0)
                    if g.bias is not None:
                        g.bias.copy_(b0)
                    self._revive(g)
                    fail_streak += 1
                    if fail_streak >= self.max_consecutive_fails:
                        if self.verbose:
                            print(f"  step {k+1}: non-finite W, {fail_streak} fails, stopping")
                        break
                    continue
                fail_streak = 0  # clean step

                gbar_run.mul_(0.9).add_(gW, alpha=0.1)  # lr-free support diagnostic
                if k < explore:
                    if self.revive_every and (k + 1) % self.revive_every == 0:
                        self._revive(g)
                else:
                    # averaging phase: accumulate Polyak iterate + tail-mean gradient
                    Wsum += g.wz
                    if bsum is not None:
                        bsum += g.bias
                    gsum += gW
                    navg += 1

            if self.verbose and (k + 1) % max(1, self.steps // 10) == 0:
                # supp = the actual lr-FREE criterion (|ḡ|≤λ1 ⇒ zero); iter-zeros is the
                # lr-dependent iterate count (sticky as lr decays) — shown only as a diagnostic
                supp_zeros = float((gbar_run.abs() <= self.l1).float().mean())
                iter_zeros = float((g.wz.abs() < 1e-8).float().mean())
                live = int(((gbar_run.abs() > self.l1).sum(0) > 0).sum())
                print(
                    f"  step {k+1:4d}/{self.steps}  loss={float(loss):+.4f}"
                    f"  λ1={l1_k:.4g}  lr={lr:.3f}  live={live}/{g.q}"
                    f"  supp-zeros={supp_zeros:.0%}  (iter-zeros={iter_zeros:.0%})"
                )
            self.history["loss"].append(float(loss))
            self.history["l1"].append(l1_k)

        navg = max(1, navg)
        W_avg = Wsum / navg
        gbar = gsum / navg
        # discovered support: KKT on the de-noised mean gradient (|ḡ_j| > λ₁ ⇒ kept)
        mask = gbar.abs() > self.l1
        W_est = W_avg * mask

        est = copy.deepcopy(g)
        with torch.no_grad():
            est.wz.copy_(W_est)
            if est.bias is not None and bsum is not None:
                est.bias.copy_(bsum / navg)

        self.model = est
        self.mask_ = mask
        self.gbar_ = gbar
        if self.verbose:
            live = int((W_est.norm(dim=0) > 1e-8).sum())
            print(
                f"[done]  live factors {live}/{g.q}  "
                f"mask zeros {float((~mask).float().mean()):.0%}  "
                f"(→ freeze mask, refit with ZQEAutoFitter for the precise fit)"
            )
        return self
