"""Diagnostic plots for ``ZQEAutoFitter`` — *see* how the fit behaves.

Every function takes a fitted ``ZQEAutoFitter`` and reads its ``.history`` /
results; nothing here re-runs the fit.  Pass an existing ``ax`` to compose, or
omit it to get a fresh figure.

The headline check is :func:`plot_objective`: the ZQE loss is ``-(m1 - m2)``,
the (negated) empirical estimating equation.  At a root ``m1 = m2``, so the
objective must **fluctuate about 0** once the refinement has converged — a
biased optimiser would sit off-zero.

Functions
---------
- :func:`plot_objective`    — ZQE objective over warm-up + refinement (0 = target).
- :func:`plot_lr`           — learning-rate schedule (anneal, then per-restart refine).
- :func:`plot_gradnorm`     — gradient norm (should decay).
- :func:`plot_convergence`  — per-restart Procrustes change + avg-grad norm vs ``tol``.
- :func:`plot_grad_balance` — tail-averaged loading gradient (estimating equation ≈ 0).
- :func:`plot_params`       — trajectories of a random selection of loadings.
- :func:`plot_deviance`     — Poisson deviance / observation over iterations.
- :func:`plot_diagnostics`  — all of the above in one dashboard.
"""
from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

from gllvm.autofit import orthogonal_align

__all__ = [
    "plot_objective", "plot_lr", "plot_gradnorm", "plot_convergence",
    "plot_grad_balance", "plot_params", "plot_deviance", "plot_diagnostics",
]


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _ax(ax, **kw):
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(**kw)
    return fig, ax


def _refine_geom(h):
    """(n_warmup_epochs, n_restarts, steps_per_restart) from a fitter history."""
    W = len(h["warmup_loss"])
    rl = h["refine_loss"]
    if not rl:
        return W, 0, 0
    return W, len(rl), rl[0].shape[-1]


def _snapshots(fitter):
    """Aligned (x, W_aligned, ref) param-snapshot trajectory.

    Warm-up: one snapshot per epoch (dense).  Refine: per-step raw SGD iterate
    from ``history["refine_wz"]`` — dense within each restart so the bump-and-
    settle pattern is visible.  All rotated into the final-model gauge (O(q)).
    Polyak consensus points (``round_wz``) are NOT included here; ``plot_params``
    overlays them separately as markers.
    """
    h = fitter.history
    W, n_rounds, steps = _refine_geom(h)
    snaps = list(h["warmup_wz"])
    xs = list(range(W))

    refine_traces = h.get("refine_wz", [])
    for r in range(n_rounds):
        if r < len(refine_traces) and refine_traces[r]:
            for k, wz_k in enumerate(refine_traces[r]):   # one entry per step
                snaps.append(wz_k)
                xs.append(W + r * steps + k)
        else:                                              # fallback: single Polyak point
            snaps.append(h["round_wz"][r])
            xs.append(W + (r + 1) * steps - 1)

    ref = fitter.model.wz.detach().to("cpu", torch.float64)
    aligned = []
    for Wt in snaps:
        Wt = Wt.to(torch.float64)
        aligned.append((Wt @ orthogonal_align(ref, Wt)).numpy())
    return np.asarray(xs), np.stack(aligned), ref


# --------------------------------------------------------------------------- #
# 1. objective  (the critical mean-0 check)
# --------------------------------------------------------------------------- #
def plot_objective(fitter, ax=None, zoom=True):
    """ZQE objective ``-(m1-m2)`` over the whole run; 0 is the target root.

    Warm-up loss (one curve) descends; in refinement each head's per-step trace
    is drawn faintly with the head-mean bold — at convergence these scatter
    about 0.  ``zoom`` crops the y-axis to the convergence region (the early
    warm-up value is annotated instead of dominating the scale).
    """
    h = fitter.history
    fig, ax = _ax(ax, figsize=(7, 4))
    W, n_rounds, steps = _refine_geom(h)

    wl = np.asarray(h["warmup_loss"], dtype=float)
    ax.plot(np.arange(W), wl, color="tab:blue", lw=1.3, label="warm-up (Adam)")

    refine_all = []
    for r in range(n_rounds):
        x0 = W + r * steps
        seg = h["refine_loss"][r]                  # (steps,)  one chain per restart
        ax.plot(np.arange(x0, x0 + steps), seg, color="tab:red", lw=0.8, alpha=0.85)
        if r:                                       # restart boundary
            ax.axvline(x0 - 0.5, color="tab:red", ls=":", lw=0.5, alpha=0.4)
        refine_all.append(seg)
    if n_rounds:
        ax.plot([], [], color="tab:red", lw=1.0, label="refine (SGD+Polyak)")

    ax.axhline(0.0, color="k", ls="--", lw=0.9, alpha=0.7)        # the target
    if W:
        ax.axvline(W - 0.5, color="gray", ls=":", lw=0.8)         # phase divider

    if refine_all:
        rv = np.concatenate([s.ravel() for s in refine_all])
        m, s = float(rv.mean()), float(rv.std())
        ax.text(0.98, 0.96, f"refine objective: mean={m:+.3g}  std={s:.2g}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
        if zoom:
            pad = 4 * s + 1e-9
            lo, hi = m - pad, m + pad
            tail = wl[W // 2:] if W else np.array([])
            if tail.size:                                          # keep visible warm-up tail
                hi = max(hi, np.percentile(tail, 90))
            ax.set_ylim(lo, hi)
            if W and wl[0] > hi:
                ax.annotate(f"warm-up starts at {wl[0]:+.1f}",
                            xy=(0, hi), xytext=(0.02, 0.9),
                            textcoords="axes fraction", fontsize=8, color="tab:blue")

    ax.set_xlabel("iteration  (warm-up epochs | refine steps)")
    ax.set_ylabel(r"ZQE objective $-(m_1-m_2)$")
    ax.set_title("Objective → 0 and fluctuating about 0 at convergence")
    ax.legend(fontsize=8, loc="lower right")
    return ax


# --------------------------------------------------------------------------- #
# 2. learning rate
# --------------------------------------------------------------------------- #
def plot_lr(fitter, ax=None):
    """Learning-rate schedule: warm-up anneal (per epoch) then refine (per round)."""
    h = fitter.history
    fig, ax = _ax(ax, figsize=(7, 3.2))
    W, n_rounds, steps = _refine_geom(h)

    ax.plot(np.arange(W), h["warmup_lr"], color="tab:blue", lw=1.3,
            label="warm-up (annealed)")
    sched = h.get("refine_lr", [])
    for r in range(n_rounds):
        x0 = W + r * steps
        if r < len(sched) and sched[r] is not None:     # within-chain Ruppert–Polyak decay
            ax.plot(np.arange(x0, x0 + steps), sched[r], color="tab:red", lw=1.3,
                    label="refine (within-chain decay)" if r == 0 else None)
        else:                                           # fallback: flat per-restart LR
            ax.plot([x0, x0 + steps - 1], [h["round_lr"][r]] * 2,
                    color="tab:red", lw=2.0, label="refine" if r == 0 else None)
    if W:
        ax.axvline(W - 0.5, color="gray", ls=":", lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("iteration  (warm-up epochs | refine steps)")
    ax.set_ylabel("learning rate")
    ax.set_title("Learning-rate schedule")
    ax.legend(fontsize=8)
    return ax


# --------------------------------------------------------------------------- #
# 3. gradient norm
# --------------------------------------------------------------------------- #
def plot_gradnorm(fitter, ax=None):
    """Gradient norm over the run (log scale); should decay toward the floor."""
    h = fitter.history
    fig, ax = _ax(ax, figsize=(7, 3.2))
    W, n_rounds, steps = _refine_geom(h)

    ax.semilogy(np.arange(W), np.asarray(h["warmup_gnorm"]) + 1e-12,
                color="tab:blue", lw=1.2, label="warm-up")
    for r in range(n_rounds):
        x0 = W + r * steps
        ax.semilogy(np.arange(x0, x0 + steps), h["refine_gnorm"][r] + 1e-12,
                    color="tab:red", lw=1.0, label="refine" if r == 0 else None)
    if W:
        ax.axvline(W - 0.5, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("iteration")
    ax.set_ylabel("grad norm")
    ax.set_title("Gradient norm")
    ax.legend(fontsize=8)
    return ax


# --------------------------------------------------------------------------- #
# 4. convergence — sequential-restart stability
# --------------------------------------------------------------------------- #
def plot_convergence(fitter, ax=None):
    """Per-restart Procrustes change and averaged-gradient norm vs ``tol``.

    Both should fall below ``tol``: the estimate stops moving across restarts
    (change), and the tail-averaged gradient reaches ≈0 (the estimating
    equation).  The first restart has no predecessor, so its change is absent.
    """
    h = fitter.history
    fig, ax = _ax(ax, figsize=(7, 3.2))
    change = np.array(h["round_change"], dtype=float)   # round 1 is nan
    rounds = np.arange(1, len(change) + 1)
    ax.semilogy(rounds, change, "o-", color="tab:purple", lw=1.5, label="restart change")
    gn = h.get("round_grad_norm", [])
    lr = h.get("round_lr", [])
    if gn:                                              # estimating-equation balance
        ax.semilogy(rounds, gn, "s--", color="tab:green", lw=1.3,
                    label=r"$\|$avg $\nabla W\|/\|W\|$")
    if gn and len(lr) == len(gn):                       # per-step drift of the iterate
        drift = np.asarray(lr, float) * np.asarray(gn, float)
        ax.semilogy(rounds, drift, "^:", color="tab:orange", lw=1.3,
                    label=r"lr$\cdot\|$avg $\nabla W\|/\|W\|$")
    ax.axhline(fitter.tol, color="gray", ls="--", lw=1.0, label=f"tol={fitter.tol}")
    ax.set_xlabel("restart")
    ax.set_ylabel("convergence diagnostics")
    status = "converged" if fitter.converged_ else "stopped"
    ax.set_title(f"Sequential-restart convergence — {status}")
    if len(rounds):
        ax.set_xticks(rounds)
        ax.set_xlim(0.5, len(rounds) + 0.5)
    ch = "—" if (fitter.change_ != fitter.change_) else f"{fitter.change_:.4f}"
    last_lr = lr[-1] if len(lr) else float("nan")
    ax.text(0.97, 0.95, f"final change = {ch}\n"
                        f"|avg ∇W|/|W| = {fitter.grad_norm_:.4f}\n"
                        f"lr·|avg ∇W|/|W| = {last_lr * fitter.grad_norm_:.2e}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
    ax.legend(fontsize=8, loc="lower left")
    return ax


# --------------------------------------------------------------------------- #
# 4b. averaged gradient — the estimating-equation / stationarity check
# --------------------------------------------------------------------------- #
def plot_grad_balance(fitter, ax=None):
    """Tail-averaged loading gradient ∂loss/∂W — should sit on 0 at a root.

    Each SGD step's gradient is noisy, but its average over the Polyak window
    estimates the gradient *at* the Polyak point, which must be ≈0 once
    converged (the empirical estimating equation).  Bars are the per-entry
    averaged gradient (head-0 = consensus gauge); most should be ~0.  Contrast
    with :func:`plot_gradnorm`, where the *instantaneous* norm stays elevated.
    """
    fig, ax = _ax(ax, figsize=(7, 3.4))
    g = fitter.avg_grad_wz_
    if g is None:
        ax.text(0.5, 0.5, "no gradient recorded", ha="center", va="center")
        return ax
    vals = g.flatten().cpu().numpy()
    ax.bar(np.arange(vals.size), vals, color="tab:green", alpha=0.75)
    ax.axhline(0.0, color="k", lw=0.9)
    band = float(np.abs(vals).mean())
    ax.axhline(band, color="gray", ls=":", lw=0.8)
    ax.axhline(-band, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("loading entry (flattened $W$)")
    ax.set_ylabel(r"tail-avg $\partial\,\mathrm{loss}/\partial W$")
    ax.set_title(rf"Averaged loading gradient $\approx$ 0   "
                 rf"($\|$avg $\nabla W\|/\|W\|$ = {fitter.grad_norm_:.4f})")
    return ax


# --------------------------------------------------------------------------- #
# 5. params — trajectories of a random selection of loadings
# --------------------------------------------------------------------------- #
def plot_params(fitter, g_true=None, k=12, ax=None, seed=0):
    """Trajectories of ``k`` random loading entries (aligned to a common gauge).

    Solid curves: raw SGD iterate — dense during warm-up and *within* each refine
    restart, so the bump-then-settle pattern is visible at each restart.  Filled
    circles mark the Polyak consensus at the end of each restart (what the fitter
    actually returns).  Dashed lines = true values if ``g_true`` is given.  Dotted
    red verticals mark restart boundaries.
    """
    fig, ax = _ax(ax, figsize=(7, 4))
    xs, Wal, ref = _snapshots(fitter)
    p, q = ref.shape
    rng = np.random.default_rng(seed)
    flat = rng.choice(p * q, size=min(k, p * q), replace=False)
    ii, jj = np.unravel_index(flat, (p, q))

    true_al = None
    if g_true is not None:
        Wt = g_true.wz.detach().to("cpu", torch.float64)
        true_al = (Wt @ orthogonal_align(ref, Wt)).numpy()

    cmap = plt.get_cmap("tab20")
    for c, (i, j) in enumerate(zip(ii, jj)):
        col = cmap(c % 20)
        ax.plot(xs, Wal[:, i, j], color=col, lw=0.9, alpha=0.85)
        if true_al is not None:
            ax.axhline(true_al[i, j], color=col, ls="--", lw=0.8, alpha=0.6)

    # Polyak consensus dots — one per restart, overlaid on the raw-iterate curve
    h = fitter.history
    W = len(h["warmup_loss"])
    _, n_rounds, steps = _refine_geom(h)
    for r, rwz in enumerate(h.get("round_wz", [])):
        x_r = W + (r + 1) * steps - 1
        Wt_r = rwz.to(torch.float64)
        Wal_r = (Wt_r @ orthogonal_align(ref, Wt_r)).numpy()
        for c, (i, j) in enumerate(zip(ii, jj)):
            ax.plot(x_r, Wal_r[i, j], "o", color=cmap(c % 20),
                    ms=5, zorder=5, markeredgewidth=0)

    # Phase / restart boundaries
    if W:
        ax.axvline(W - 0.5, color="gray", ls=":", lw=0.8)
    for r in range(1, n_rounds):
        ax.axvline(W + r * steps - 0.5, color="tab:red", ls=":", lw=0.6, alpha=0.5)

    ax.set_xlabel("iteration  (warm-up epochs | refine steps)")
    ax.set_ylabel("loading value (aligned)")
    tgt = "; dashed = true" if g_true is not None else ""
    ax.set_title(f"Loading trajectories — {len(flat)} random entries{tgt}")
    return ax


# --------------------------------------------------------------------------- #
# 6. deviance — goodness of fit over iterations  (Poisson log-link)
# --------------------------------------------------------------------------- #
def _poisson_deviance_per_obs(model, y, z):
    """Mean Poisson deviance per observation for a log-link decoder."""
    eta = model.forward(z).clamp(max=10.0)
    mu = torch.exp(eta)
    yf = y.to(mu.dtype)
    term = torch.special.xlogy(yf, yf / mu) - (yf - mu)            # 0·log0 = 0
    return float(2.0 * term.sum(-1).mean())


def plot_deviance(fitter, ax=None, max_points=120):
    """Mean Poisson deviance / observation over iterations (lower = better fit).

    Recomputed post-hoc from the recorded param snapshots, using the fitter's
    own encoder and data.  Assumes a Poisson **log-link** decoder (the setting
    of this experiment).
    """
    h = fitter.history
    fig, ax = _ax(ax, figsize=(7, 3.2))
    y = fitter.y_
    W, n_rounds, steps = _refine_geom(h)

    template = copy.deepcopy(fitter.model)
    enc = fitter.encoder_factory(template)

    def dev_at(wz, bias):
        with torch.no_grad():
            template.wz.copy_(wz.to(template.wz.dtype).to(template.wz.device))
            if bias is not None and template.bias is not None:
                template.bias.copy_(bias.to(template.bias.dtype).to(template.bias.device))
            z, _, _ = enc.sample(y)
            return _poisson_deviance_per_obs(template, y, z)

    # warm-up: thin to <= max_points
    idx = np.unique(np.linspace(0, W - 1, min(W, max_points)).astype(int)) if W else []
    xs_w = list(idx)
    dv_w = [dev_at(h["warmup_wz"][t], h["warmup_bias"][t]) for t in idx]
    ax.plot(xs_w, dv_w, color="tab:blue", lw=1.3, label="warm-up")

    # refine: one point per round (consensus), at round end
    xs_r, dv_r = [], []
    for r in range(n_rounds):
        xs_r.append(W + (r + 1) * steps - 1)
        dv_r.append(dev_at(h["round_wz"][r], h["round_bias"][r]))
    if xs_r:
        ax.plot(xs_r, dv_r, "o-", color="tab:red", lw=1.3, label="refine consensus")

    if W:
        ax.axvline(W - 0.5, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("iteration  (warm-up epochs | refine steps)")
    ax.set_ylabel("Poisson deviance / obs")
    ax.set_title("Deviance (goodness of fit) over iterations")
    ax.legend(fontsize=8)
    return ax


# --------------------------------------------------------------------------- #
# dashboard
# --------------------------------------------------------------------------- #
def plot_diagnostics(fitter, g_true=None, title=None, figsize=(19, 8)):
    """2×4 dashboard: objective, LR, grad-norm, grad-balance / heads, params, deviance."""
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    plot_objective(fitter,     ax=axes[0, 0])
    plot_lr(fitter,            ax=axes[0, 1])
    plot_gradnorm(fitter,      ax=axes[0, 2])
    plot_grad_balance(fitter,  ax=axes[0, 3])
    plot_convergence(fitter,   ax=axes[1, 0])
    plot_params(fitter, g_true=g_true, ax=axes[1, 1])
    try:
        plot_deviance(fitter,  ax=axes[1, 2])
    except Exception as e:                                          # non-Poisson etc.
        axes[1, 2].text(0.5, 0.5, f"deviance n/a\n({e})", ha="center",
                        va="center", fontsize=8)
    axes[1, 3].axis("off")                                          # spare panel
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig
