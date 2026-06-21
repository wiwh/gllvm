"""
Automatic ZQE fitter for GLLVMs.

``ZQEAutoFitter`` wraps the ZQE estimating-equation update in a robust,
low-variance training recipe with a small, explicit API.

Recipe
------
1. **Warm-up (Adam).**  A fast optimiser drives the decoder into the basin of
   the ZQE solution.  Learning rate is annealed by a grad-norm plateau
   scheduler (``ReduceLROnPlateau`` on the per-epoch mean gradient norm); the
   phase exits once the LR hits a floor (we are "close").

2. **Refinement (SGD + Polyak, single chain).**  From the warm-started decoder
   we run constant-LR SGD and maintain a Polyak–Ruppert average (uniform mean of
   the tail iterates).  Constant-LR SGD + Polyak averaging is the statistically
   efficient estimator of the ZQE solution and the source of variance reduction
   — the right choice when the *parameters themselves* are the object of inference.

3. **Sequential-restart convergence check.**  Each round warm-restarts the chain
   from the current Polyak estimate and re-runs SGD+Polyak.  We Procrustes-align
   the new estimate to the previous one (loadings carry an O(q) rotation
   ambiguity) and measure how far it moved:
     * change < ``tol``               → converged, return the estimate;
     * otherwise                       → restart again, escalating the LR after
                                         repeated stalls (probes basin robustness
                                         — a higher-LR restart that lands
                                         elsewhere reveals a non-robust optimum).
   A complementary check is the **tail-averaged gradient** ``grad_norm_`` =
   ‖avg ∇W‖/‖W‖: the ZQE root is ∇_θ[-(m1-m2)]=0, so this must be ≈0 at
   convergence (the per-step gradient stays noisy; its average cancels to ≈0).

Note: the cross-head "uncertainty" of earlier versions was dropped — heads share
one dataset, so their spread is purely *algorithmic* noise, not a standard error.

The fitter is **family-agnostic**: it only uses ``gllvm.sample``,
``gllvm.zq_log`` and a parameter-free ``encoder.sample`` interface.  By default
it uses the closed-form Gaussian-log1p MAP encoder, which holds a live
reference to the decoder and therefore tracks θ automatically.
"""

from __future__ import annotations

import copy
from typing import Callable, Optional

import numpy as np
import torch
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel

from gllvm.encoder import MapEncoderGaussianLog1p

__all__ = ["ZQEAutoFitter", "orthogonal_align", "procrustes_error"]


_OPTIMS = {
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "sgd": optim.SGD,
    "rmsprop": optim.RMSprop,
}


def _make_opt(spec, params, lr):
    """Build an optimiser from a string key or an optimiser class."""
    cls = _OPTIMS[spec.lower()] if isinstance(spec, str) else spec
    return cls(params, lr=lr)


def orthogonal_align(ref: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """
    Orthogonal Procrustes rotation aligning ``W`` to ``ref``.

    Returns the orthogonal ``R`` (q×q, reflections allowed) minimising
    ``||ref - W @ R||_F``.  This is the loadings' O(q) gauge freedom:
    ``W`` and ``W @ R`` describe the same model when ``z ~ N(0, I)``.
    """
    M = ref.T @ W                                  # (q, q)
    U, _, Vt = torch.linalg.svd(M)
    R1 = (U @ Vt).T                                # proper rotation
    D = torch.eye(U.shape[0], device=ref.device, dtype=ref.dtype)
    D[-1, -1] = -1.0
    R2 = (U @ D @ Vt).T                            # reflection alternative
    e1 = torch.linalg.norm(ref - W @ R1)
    e2 = torch.linalg.norm(ref - W @ R2)
    return R1 if e1 <= e2 else R2


def procrustes_error(W_true, W_est) -> float:
    """Relative orthogonal Procrustes error between two loading matrices.

    ``min_{R in O(q)} ||W_true - W_est @ R||_F / ||W_true||_F`` — the metric
    used throughout the simulations (pure rotation, no centring/rescaling; see
    the project notes).  ``R`` is the optimal rotation from ``orthogonal_align``
    (reflections allowed).  Accepts numpy arrays or torch tensors for either
    argument, so it can compare a Python ``GLLVM.wz`` against R's loadings.
    """
    def _to64(M):
        if isinstance(M, torch.Tensor):
            return M.detach().to("cpu", torch.float64)
        return torch.as_tensor(np.asarray(M), dtype=torch.float64)

    Wt, We = _to64(W_true), _to64(W_est)
    R = orthogonal_align(Wt, We)
    return float(torch.linalg.norm(Wt - We @ R) / torch.linalg.norm(Wt))


class ZQEAutoFitter:
    """
    Automatic ZQE fitter: Adam warm-up → single-chain SGD + Polyak averaging,
    with a sequential-restart convergence check.

    Parameters
    ----------
    gllvm : GLLVM
        Decoder whose parameters (wz, bias, log_scale) are estimated.  It is
        modified in place by the warm-up; the Polyak-averaged estimate is
        returned via ``.model`` (the input ``gllvm`` is left at its warm-up state).
    encoder_factory : callable, optional
        ``g -> encoder`` building a parameter-free encoder bound to decoder
        ``g``.  Defaults to ``MapEncoderGaussianLog1p``.
    device : str
        'cpu' or 'cuda'.

    Data / sampling
    ---------------
    batch_size : int or None
        Observed mini-batch size.  ``None`` → full batch (recommended when N
        is small, as the ZQE target is then a fixed deterministic root).
    sim_factor : float
        Fantasy-sample count = ``round(sim_factor * batch_size)``.

    Warm-up (Adam)
    --------------
    warmup_optimizer : str or optimiser class      (default 'adam')
    warmup_lr : float
    warmup_max_epochs : int
    warmup_factor, warmup_patience, warmup_threshold : ReduceLROnPlateau knobs
        (plateau is detected on the per-epoch mean gradient norm).
    warmup_min_lr : float
        Warm-up exits once the scheduled LR reaches this floor.

    Refinement (SGD + Polyak, sequential restarts)
    ----------------------------------------------
    refine_optimizer : str or optimiser class       (default 'sgd')
    refine_lr : float
        LR for the first chain (held constant *within* a restart so the Polyak
        tail average is valid).
    refine_lr_power : float
        Within-chain Ruppert–Polyak decay exponent: step ``k`` uses
        ``lr/(1+k)**power`` (``0`` → constant LR).  A decreasing LR + tail
        averaging is the √n-optimal estimator and removes the O(lr) noise floor
        at its source; ``0.5`` is the classic choice.
    refine_lr_decay : float
        Multiplier applied to the *base* LR between successive restarts
        (``<1`` anneals further).  Complements the within-chain decay.
    steps_per_round : int
        SGD epochs per restart.
    max_rounds : int
        Maximum number of restarts.
    ema_decay : float or None
        If ``None`` (default) a uniform Polyak–Ruppert tail average is used.
        If a float in (0,1), an exponential moving average is used instead
        (``avg <- ema_decay*avg + (1-ema_decay)*cur``).
    polyak_warmup_frac : float
        Fraction of each restart's steps to skip before averaging begins.

    Restart / stopping
    ------------------
    Each round warm-restarts the chain from the current Polyak estimate and
    re-runs SGD+Polyak at the **decayed** LR (``refine_lr * refine_lr_decay**r``);
    convergence is declared once the new estimate barely moves from the previous
    one (Procrustes-aligned change < ``tol``).  Watch ``lr·‖avg ∇W‖/‖W‖`` (the
    per-step drift of the iterate) shrink alongside the change.
    tol : float
        Convergence target on the relative Procrustes change between restarts.

    Misc
    ----
    grad_clip : float
        Max gradient norm (per step).
    seed : int or None
        If given, ``torch.manual_seed(seed)`` is set at the start of ``fit``.
    verbose : bool
    """

    def __init__(
        self,
        gllvm,
        *,
        encoder_factory: Optional[Callable] = None,
        device: str = "cpu",
        # data / sampling
        batch_size: Optional[int] = None,
        sim_factor: float = 1.0,
        # warm-up
        warmup_optimizer="adam",
        warmup_lr: float = 0.1,
        warmup_max_epochs: int = 1500,
        warmup_factor: float = 0.5,
        warmup_patience: int = 40,
        warmup_threshold: float = 1e-3,
        warmup_min_lr: float = 2e-3,
        # refinement
        refine_optimizer="sgd",
        refine_lr: float = 0.3,
        refine_lr_decay: float = 0.5,
        refine_lr_power: float = 0.5,
        steps_per_round: int = 200,
        max_rounds: int = 8,
        ema_decay: Optional[float] = None,
        polyak_warmup_frac: float = 0.2,
        # restart / stopping
        tol: float = 0.02,
        # misc
        grad_clip: float = 5.0,
        seed: Optional[int] = None,
        verbose: bool = True,
    ):
        self.gllvm = gllvm.to(device)
        self.device = device
        self.encoder_factory = encoder_factory or (lambda g: MapEncoderGaussianLog1p(g))

        self.batch_size = batch_size
        self.sim_factor = sim_factor

        self.warmup_optimizer = warmup_optimizer
        self.warmup_lr = warmup_lr
        self.warmup_max_epochs = warmup_max_epochs
        self.warmup_factor = warmup_factor
        self.warmup_patience = warmup_patience
        self.warmup_threshold = warmup_threshold
        self.warmup_min_lr = warmup_min_lr

        self.refine_optimizer = refine_optimizer
        self.refine_lr = refine_lr
        self.refine_lr_decay = refine_lr_decay
        self.refine_lr_power = refine_lr_power
        self.steps_per_round = steps_per_round
        self.max_rounds = max_rounds
        self.ema_decay = ema_decay
        self.polyak_warmup_frac = polyak_warmup_frac

        self.tol = tol

        self.grad_clip = grad_clip
        self.seed = seed
        self.verbose = verbose

        # ---- results (filled by fit) ----
        self.model: Optional["torch.nn.Module"] = None   # Polyak-averaged estimate
        self.change_: float = float("nan")                # Procrustes change at last restart
        self.grad_norm_: float = float("nan")             # ||tail-avg ∇W|| / ||W|| (≈0 at root)
        self.avg_grad_wz_: Optional[torch.Tensor] = None  # (p,q) tail-averaged loading gradient
        self.converged_: bool = False
        self.n_rounds_used_: int = 0
        self.y_: Optional[torch.Tensor] = None            # data (for post-hoc deviance)
        self.history = {
            "warmup_loss": [], "warmup_gnorm": [], "warmup_lr": [],
            "warmup_wz": [], "warmup_bias": [],            # per-epoch param snapshots
            "round_change": [], "round_lr": [], "round_grad_norm": [],
            "refine_loss": [], "refine_gnorm": [], "refine_lr": [],   # per restart: (steps,) arrays
            "refine_wz": [],                               # per restart: list of per-step raw-iterate (p,q) snapshots
            "round_wz": [], "round_bias": [],              # per-restart Polyak snapshots
        }

    # ------------------------------------------------------------------
    # one ZQE gradient step (family-agnostic; encoder detached = exact)
    # ------------------------------------------------------------------
    def _zqe_step(self, gllvm, encoder, opt, y_batch, n_sim):
        """Single ZQE update: loss = -(m1 - m2).

        Returns ``(loss, grad_norm, wz_grad)`` — ``wz_grad`` is the *unclipped*
        loading gradient ∂loss/∂W (the estimating function in loadings space),
        so its tail average can be tracked as a stationarity check.
        """
        with torch.no_grad():
            yq = gllvm.sample(z=gllvm.sample_z(n_sim))
            z, _, _ = encoder.sample(y_batch)          # detached: score-fn identity
            z_q, _, _ = encoder.sample(yq)

        m1 = gllvm.zq_log(y_batch, z=z).sum(-1).mean()
        m2 = gllvm.zq_log(yq, z=z_q).sum(-1).mean()
        loss = -(m1 - m2)

        opt.zero_grad()
        loss.backward()
        wz_grad = gllvm.wz.grad.detach().clone()       # before clipping = true gradient
        gn = torch.nn.utils.clip_grad_norm_(gllvm.parameters(), self.grad_clip).item()
        if torch.isfinite(loss):
            opt.step()
        return loss.item(), gn, wz_grad

    def _epoch(self, gllvm, encoder, opt, y, bs, n_sim):
        """One pass over the data; returns (mean_loss, mean_grad_norm, mean_wz_grad)."""
        n = len(y)
        if bs >= n:
            return self._zqe_step(gllvm, encoder, opt, y, n_sim)
        perm = torch.randperm(n, device=self.device)
        losses, gns, grads = [], [], []
        for s in range(0, n, bs):
            yb = y[perm[s:s + bs]]
            ns = max(1, round(self.sim_factor * len(yb)))
            l, g, wzg = self._zqe_step(gllvm, encoder, opt, yb, ns)
            losses.append(l); gns.append(g); grads.append(wzg)
        return float(np.mean(losses)), float(np.mean(gns)), torch.stack(grads).mean(0)

    # ------------------------------------------------------------------
    # Phase 1 — warm-up
    # ------------------------------------------------------------------
    def _warmup(self, y, bs, n_sim):
        g = self.gllvm
        enc = self.encoder_factory(g)
        opt = _make_opt(self.warmup_optimizer, g.parameters(), self.warmup_lr)
        sched = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=self.warmup_factor,
            patience=self.warmup_patience, threshold=self.warmup_threshold,
            min_lr=self.warmup_min_lr,
        )
        if self.verbose:
            print(f"[warm-up] {self.warmup_optimizer} lr={self.warmup_lr} "
                  f"(exit at lr≤{self.warmup_min_lr})")
        for ep in range(self.warmup_max_epochs):
            loss, gn, _ = self._epoch(g, enc, opt, y, bs, n_sim)
            sched.step(gn)
            lr = opt.param_groups[0]["lr"]
            self.history["warmup_loss"].append(loss)
            self.history["warmup_gnorm"].append(gn)
            self.history["warmup_lr"].append(lr)
            self.history["warmup_wz"].append(g.wz.detach().cpu().clone())
            self.history["warmup_bias"].append(
                None if g.bias is None else g.bias.detach().cpu().clone())
            if self.verbose and (ep + 1) % 100 == 0:
                print(f"  ep {ep+1:4d}  loss={loss:+.4f}  gnorm={gn:.4f}  lr={lr:.2e}")
            if lr <= self.warmup_min_lr * 1.0001 and ep >= self.warmup_patience:
                if self.verbose:
                    print(f"  warm-up done at ep {ep+1} (lr floor reached)")
                break

    # ------------------------------------------------------------------
    # Phase 2 — single-chain refinement (sequential restarts)
    # ------------------------------------------------------------------
    def _run_chain(self, base, y, bs, lr):
        """Warm-restart a single chain from ``base``, run SGD + Polyak.

        The LR decays *within* the chain (Ruppert–Polyak: ``lr_k = lr/(1+k)**power``)
        and the tail iterates are averaged — this is the √n-optimal estimator and
        removes the O(lr) noise floor of constant-LR averaging.  Returns
        ``(averaged_decoder, losses, gnorms, avg_wz_grad, lr_sched)``: per-step ZQE
        objective and grad-norm traces, the loading gradient averaged over the
        Polyak window (≈0 at a root), and the per-step LR schedule.
        """
        g = copy.deepcopy(base).to(self.device)
        enc = self.encoder_factory(g)
        opt = _make_opt(self.refine_optimizer, g.parameters(), lr)

        avg_fn = None
        if self.ema_decay is not None:
            d = float(self.ema_decay)
            avg_fn = lambda avg, cur, n: d * avg + (1.0 - d) * cur
        ema = AveragedModel(g, avg_fn=avg_fn)        # default avg_fn = uniform mean

        n_sim = max(1, round(self.sim_factor * min(bs, len(y))))
        K = self.steps_per_round
        start = int(self.polyak_warmup_frac * K)
        # within-chain Ruppert–Polyak schedule: lr_k = lr / (1+k)^power (power=0 → constant).
        lr_sched = lr / np.power(1.0 + np.arange(K), self.refine_lr_power)
        losses, gnorms = [], []
        gbar = torch.zeros_like(g.wz); gcount = 0        # tail-averaged ∇W (Polyak window)
        wz_trace = []                                      # raw SGD iterate per step (for diagnostics)
        for k in range(K):
            for pg in opt.param_groups:
                pg["lr"] = float(lr_sched[k])
            l, gnorm, wzg = self._epoch(g, enc, opt, y, bs, n_sim)
            losses.append(l); gnorms.append(gnorm)
            wz_trace.append(g.wz.detach().cpu().clone())  # raw iterate (before Polyak average)
            if k >= start:
                ema.update_parameters(g)
                if torch.isfinite(wzg).all():
                    gbar += wzg; gcount += 1
        gbar = (gbar / max(1, gcount)).detach()
        return ema.module, np.asarray(losses), np.asarray(gnorms), gbar, lr_sched, wz_trace

    def _refine(self, y, bs):
        """Single chain, warm-restarted each round at a decayed LR; converge when
        the Polyak estimate stops moving (Procrustes-aligned change < tol)."""
        est = copy.deepcopy(self.gllvm).to(self.device)   # start = warm-up result
        lr = self.refine_lr

        start = int(self.polyak_warmup_frac * self.steps_per_round)
        for rnd in range(self.max_rounds):
            new, losses, gnorms, gbar, lr_sched, wz_trace = self._run_chain(est, y, bs, lr)

            # restart stability: how far did the (gauge-aligned) Polyak estimate move?
            change = float("nan") if rnd == 0 else procrustes_error(est.wz, new.wz)
            grad_norm = float(gbar.norm() / (new.wz.detach().norm() + 1e-12))
            lr_eff = float(lr_sched[start:].mean() if len(lr_sched) > start
                           else lr_sched.mean())         # mean LR over the averaged tail
            drift = lr_eff * grad_norm                   # per-step drift of the iterate

            self.history["round_change"].append(change)
            self.history["round_lr"].append(lr_eff)      # effective (tail-mean) LR
            self.history["refine_lr"].append(lr_sched)   # per-step within-chain schedule
            self.history["round_grad_norm"].append(grad_norm)
            self.history["refine_wz"].append(wz_trace)               # list of per-step (p,q) tensors
            self.history["refine_loss"].append(losses)               # (steps,)
            self.history["refine_gnorm"].append(gnorms)
            self.history["round_wz"].append(new.wz.detach().cpu().clone())
            self.history["round_bias"].append(
                None if new.bias is None else new.bias.detach().cpu().clone())

            self.model = new
            self.change_ = change
            self.grad_norm_ = grad_norm
            self.avg_grad_wz_ = gbar.detach().cpu().clone()
            self.n_rounds_used_ = rnd + 1
            if self.verbose:
                ch = "—" if change != change else f"{change:.4f}"
                print(f"[refine] restart {rnd+1}/{self.max_rounds}  change={ch}  "
                      f"|avg∇W|/|W|={grad_norm:.4f}  lr·|avg∇W|/|W|={drift:.2e}  "
                      f"lr0={lr:.2e} lr_eff={lr_eff:.2e}  (tol={self.tol})")

            est = new                                    # warm-restart from current estimate
            if rnd >= 1 and change < self.tol:
                self.converged_ = True
                if self.verbose:
                    print(f"  ✓ converged (restart change {change:.4f} < tol {self.tol})")
                break
            lr *= self.refine_lr_decay                   # anneal base LR between restarts too

        if self.verbose and not self.converged_:
            print(f"[refine] stopped at max_rounds={self.max_rounds} "
                  f"(change={self.change_:.4f})")

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def fit(self, y: torch.Tensor):
        """Run warm-up + sequential-restart refinement.  Returns ``self``."""
        if self.seed is not None:
            torch.manual_seed(self.seed)
        y = y.to(self.device)
        self.y_ = y                                  # kept for post-hoc deviance plots
        bs = len(y) if self.batch_size is None else int(self.batch_size)
        n_sim = max(1, round(self.sim_factor * min(bs, len(y))))

        self._warmup(y, bs, n_sim)
        self._refine(y, bs)
        return self

    def summary(self) -> str:
        msg = (
            f"ZQEAutoFitter: {'converged' if self.converged_ else 'stopped'} "
            f"after {self.n_rounds_used_} restart(s); "
            f"restart change = {self.change_:.4f} (tol {self.tol}); "
            f"|avg ∇W|/|W| = {self.grad_norm_:.4f}."
        )
        if self.verbose:
            print(msg)
        return msg
