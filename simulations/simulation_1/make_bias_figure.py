"""Loading bias diagnostic for simulation 1.

Pools every loading element over reps and response dimensions.  To avoid
sign-confounding (positive and negative loadings have opposite-sign residuals
that cancel in a naive mean), we plot the *sign-corrected residual*:

    signed_resid = sign(W_true) × (W_est − W_true)

vs the loading magnitude |W_true|.  This is always negative under shrinkage
(estimate pulled toward zero) regardless of the sign of the true loading, so
ridge and unpenalized biases are comparable without cancellation.

Writes ``paper/figures/sim1_bias.pdf``.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sweep  # noqa: E402

OUT = os.path.abspath(os.path.join(HERE, "..", "..", "paper", "figures",
                                   "sim1_bias.pdf"))

METHODS = ["zqe", "gllvm"]
LABEL = {"zqe": r"$Z_q$ (ridge)", "gllvm": "gllvm (no ridge)"}
MCOLOR = {"zqe": "#1f77b4", "gllvm": "#d62728"}

plt.rcParams.update({"figure.dpi": 150, "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 11})


def binned_mean(x, y, nbins=20):
    """Mean y in equal-count (quantile) bins of x."""
    order = np.argsort(x)
    x, y = x[order], y[order]
    edges = np.quantile(x, np.linspace(0, 1, nbins + 1))
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, nbins - 1)
    bx = np.array([x[idx == b].mean() for b in range(nbins) if (idx == b).any()])
    by = np.array([y[idx == b].mean() for b in range(nbins) if (idx == b).any()])
    return bx, by


def main():
    df = sweep.load_results()
    truth = (df[df.method == "true"][["p", "n", "rep", "param", "i", "j", "value"]]
             .rename(columns={"value": "true"}))
    est = df[df.method.isin(METHODS) & (df.failed == 0.0) & (df.param == "W")]
    res = est.merge(truth, on=["p", "n", "rep", "param", "i", "j"], how="left")

    # sign-corrected residual: negative = shrinkage, positive = expansion
    res["abs_true"] = res["true"].abs()
    res["signed_resid"] = np.sign(res["true"]) * (res["value"] - res["true"])
    # drop entries very close to zero (sign undefined, numerical noise)
    res = res[res["abs_true"] > 0.02]

    N_VALS = sorted(res.n.unique())
    fig, axes = plt.subplots(1, len(N_VALS), figsize=(3.4 * len(N_VALS), 3.4),
                             sharey=True, sharex=True)
    for ax, n in zip(np.atleast_1d(axes), N_VALS):
        sub = res[res.n == n]
        for m in METHODS:
            d = sub[sub.method == m]
            x = d["abs_true"].to_numpy(float)
            y = d["signed_resid"].to_numpy(float)
            ax.scatter(x, y, s=3, alpha=0.05, color=MCOLOR[m], edgecolors="none")
            bx, by = binned_mean(x, y)
            ax.plot(bx, by, "-", color=MCOLOR[m], lw=2.0, label=LABEL[m])
            slope = np.polyfit(x, y, 1)[0]
            if m == "zqe":
                ax.text(0.04, 0.06, f"$Z_q$ slope $= {slope:+.3f}$",
                        transform=ax.transAxes, fontsize=9, color=MCOLOR[m])
        ax.axhline(0.0, color="k", lw=0.8, ls="--", alpha=0.6)
        ax.set_ylim(-0.6, 0.4)
        ax.set_title(f"$n = {n}$")
        ax.set_xlabel(r"true loading magnitude $|w_{ij}|$")
    axes[0].set_ylabel(r"sign$(w_{ij})\times(\hat{w}_{ij} - w_{ij})$"
                       "\n(negative = shrinkage)")
    axes[0].legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.suptitle(r"Sign-corrected loading bias vs. $|w_{ij}|$ (pooled over reps and $p$):"
                 " ridge shrinkage is mild and decays with $n$",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print("wrote", OUT)

    print("\nshrinkage slope (sign-corrected resid on |true|), per n:")
    for n in N_VALS:
        sub = res[res.n == n]
        row = []
        for m in METHODS:
            d = sub[sub.method == m]
            s = np.polyfit(d["abs_true"].to_numpy(float),
                           d["signed_resid"].to_numpy(float), 1)[0]
            row.append(f"{m}={s:+.4f}")
        print(f"  n={n:<4} " + "  ".join(row))


if __name__ == "__main__":
    main()
