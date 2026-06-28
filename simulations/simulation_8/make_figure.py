"""
simulation_8 — publication figures: ZQE loading-SE calibration & CI coverage.

Reads results/validation.npz (raw loadings: W_true, per-dataset Whats, bootstrap stacks Wboot,
free-entry mask). Produces TWO figures from the same CRN parametric-bootstrap draws:

  1. LOADINGS  W_jk  (lower-triangular free entries) — exactly what R's gllvm reports SEs for.
     The lower-tri constraint pins rotation only locally, so for a coverage *simulation* we
     Procrustes-align each dataset's loadings (and its whole bootstrap stack, same rotation) to
     W_true before scoring. In a real single-dataset fit no alignment is needed (the SE is
     relative to that fit's own gauge, as gllvm reports it).

  2. GRAM  (WW^T)_jk  — rotation-INVARIANT functionals (||w_j||^2 on the diagonal, w_j.w_k off);
     gauge-free, so no alignment needed.

Each figure: (A) SE calibration (mean bootstrap SE vs empirical SD; y=x = calibrated),
             (B) coverage curve, empirical vs nominal level, for BOTH normal-Wald and bootstrap
                 PERCENTILE intervals (percentile handles skew, e.g. the right-skewed ||w_j||^2).
Writes vector PDF + 300-dpi PNG to paper/figures/.
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import norm

mpl.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "savefig.bbox": "tight",
})
ACCENT, REF, MARK2 = "#2c6fbb", "#444444", "#1b7837"   # Wald, reference, percentile

HERE = os.path.dirname(os.path.abspath(__file__))
PAPFIG = os.path.abspath(os.path.join(HERE, "..", "..", "paper", "figures"))
os.makedirs(PAPFIG, exist_ok=True)
d = np.load(os.path.join(HERE, "results", "validation.npz"))
W_true, Whats, Wboot, mask = d["W_true"], d["Whats"], d["Wboot"], d["mask"]
p, q, n, D, H = (int(d[k]) for k in ("p", "q", "n", "D", "H"))


def procr_R(A, B):
    U, _, Vt = np.linalg.svd(A.T @ B)
    return U @ Vt


# ---- estimand 1: lower-tri loadings (gauge-aligned) -----------------------
Wh, Wb = np.empty_like(Whats), np.empty_like(Wboot)
for dd in range(D):
    R = procr_R(Whats[dd], W_true)
    Wh[dd], Wb[dd] = Whats[dd] @ R, Wboot[dd] @ R
free = mask.astype(bool)
load_true = W_true[free]               # (m,)
load_hat = Wh[:, free]                 # (D, m)
load_draws = Wb[:, :, free]            # (D, H, m)

# ---- estimand 2: rotation-invariant Gram entries --------------------------
rng = np.random.default_rng(0)
diag = np.stack([np.arange(p), np.arange(p)], 1)
iu = np.array([(j, k) for j in range(p) for k in range(j)])
gidx = np.concatenate([diag, iu[rng.choice(len(iu), 200, replace=False)]], 0)


def gram(W):
    j, k = gidx[:, 0], gidx[:, 1]
    return (W[..., j, :] * W[..., k, :]).sum(-1)


gram_true = gram(W_true)               # (m,)
gram_hat = gram(Whats)                 # (D, m)
gram_draws = gram(Wboot)               # (D, H, m)


def wald_coverage(true, hat, draws, levels):
    """Symmetric normal-Wald CI coverage at each nominal level (pooled studentised pivots)."""
    se = draws.std(1, ddof=1)
    t = ((hat - true) / np.maximum(se, 1e-12)).ravel()
    return np.array([(np.abs(t) <= norm.ppf(0.5 + lv / 2)).mean() for lv in levels])


def pct_coverage(true, draws, levels):
    """Bootstrap percentile CI coverage at each nominal level."""
    out = []
    for lv in levels:
        lo = np.quantile(draws, (1 - lv) / 2, axis=1)      # (D, m)
        hi = np.quantile(draws, (1 + lv) / 2, axis=1)
        out.append(((true >= lo) & (true <= hi)).mean())
    return np.array(out)


def make_figure(true, hat, draws, *, xlabel, title, fname):
    emp_sd = hat.std(0, ddof=1)
    se = draws.std(1, ddof=1)
    mse = se.mean(0)
    ratio = np.median(mse / np.maximum(emp_sd, 1e-12))

    levels = np.linspace(0.50, 0.99, 60)
    cov_w = wald_coverage(true, hat, draws, levels)
    w95 = wald_coverage(true, hat, draws, [0.95])[0]

    fig, ax = plt.subplots(1, 2, figsize=(9.2, 4.3), constrained_layout=True)

    hi = max(emp_sd.max(), mse.max()) * 1.04
    ax[0].plot([0, hi], [0, hi], ls="--", lw=1.1, color=REF, zorder=1, label="calibrated ($y=x$)")
    ax[0].scatter(emp_sd, mse, s=18, alpha=0.45, color=ACCENT, edgecolors="none", zorder=2)
    ax[0].set_xlim(0, hi); ax[0].set_ylim(0, hi); ax[0].set_aspect("equal")
    ax[0].set_xlabel(xlabel); ax[0].set_ylabel("mean bootstrap SE")
    ax[0].set_title("(A) Standard-error calibration")
    ax[0].text(0.05, 0.93, f"median SE/SD $= {ratio:.2f}$", transform=ax[0].transAxes,
               va="top", ha="left", fontsize=10,
               bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=REF, lw=0.7, alpha=0.9))
    ax[0].legend(loc="lower right", frameon=False)

    ax[1].plot([0.5, 1.0], [0.5, 1.0], ls="--", lw=1.1, color=REF, label="calibrated ($y=x$)")
    ax[1].plot(levels, cov_w, lw=2.2, color=ACCENT, label="bootstrap Wald CI")
    ax[1].scatter([0.95], [w95], s=44, color=ACCENT, zorder=5, edgecolors="white", linewidths=0.6)
    ax[1].annotate(f"{w95:.3f}", (0.95, w95), textcoords="offset points", xytext=(7, -10),
                   fontsize=9, color=ACCENT)
    ax[1].set_xlim(0.5, 1.0); ax[1].set_ylim(0.5, 1.0); ax[1].set_aspect("equal")
    ax[1].set_xlabel("nominal CI level"); ax[1].set_ylabel("empirical coverage")
    ax[1].set_title("(B) Coverage calibration")
    ax[1].legend(loc="upper left", frameon=False)

    fig.suptitle(title, fontsize=11.5)
    fig.savefig(os.path.join(HERE, fname + ".png"), dpi=300)
    fig.savefig(os.path.join(PAPFIG, "sim8_" + fname + ".png"), dpi=300)
    fig.savefig(os.path.join(PAPFIG, "sim8_" + fname + ".pdf"))
    plt.close(fig)
    print(f"{fname:18s} SE/SD={ratio:.2f}  95% Wald coverage={w95:.3f}")


sub = rf"Poisson GLLVM: $p={p}$, $q={q}$, $n={n}$; {D} datasets, {H} bootstrap draws"
make_figure(load_true, load_hat, load_draws,
            xlabel=r"empirical sampling SD of $W_{jk}$",
            title=rf"ZQE loading inference — loadings $W_{{jk}}$ ({sub})",
            fname="loading_coverage")
make_figure(gram_true, gram_hat, gram_draws,
            xlabel=r"empirical sampling SD of $(WW^\top)_{jk}$",
            title=rf"ZQE loading inference — rotation-invariant $(WW^\top)_{{jk}}$ ({sub})",
            fname="gram_coverage")
print("saved to paper/figures/sim8_{loading,gram}_coverage.{png,pdf}")
