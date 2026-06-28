"""Generate the simulation-3 paper figure: Procrustes-error boxplots (Binomial).

Reads per-rep CSVs under ``results/`` (WZS=0.5, L2=0.5/n, Binomial N=10) and
writes ``paper/figures/sim3_procrustes_box.pdf``.  Same layout as sim_1 but for
the bounded-count (Binomial) twin.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sweep  # noqa: E402

OUT = os.path.abspath(os.path.join(HERE, "..", "..", "paper", "figures",
                                   "sim3_procrustes_box.pdf"))

METHODS = ["zqe", "gllvm"]
LABEL = {"zqe": r"$Z_q$ (ridge)", "gllvm": "gllvm"}
MCOLOR = {"zqe": "#1f77b4", "gllvm": "#d62728"}

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})


def main():
    df = sweep.load_results()
    fits = (df[df.method != "true"]
            .drop_duplicates(["p", "n", "rep", "method"]))
    ok = fits[fits.failed == 0.0]
    P_VALS = sorted(ok.p.unique())
    N_VALS = sorted(ok.n.unique())

    fig, axes = plt.subplots(1, len(P_VALS),
                             figsize=(3.1 * len(P_VALS), 3.4), sharey=True)
    width, offs = 0.32, {"zqe": -0.18, "gllvm": +0.18}
    xticks = np.arange(len(N_VALS))
    for ax, p in zip(np.atleast_1d(axes), P_VALS):
        sub = ok[ok.p == p]
        for m in METHODS:
            data = [sub[(sub.n == n) & (sub.method == m)].procrustes.dropna().values
                    for n in N_VALS]
            bp = ax.boxplot(data, positions=xticks + offs[m], widths=width,
                            patch_artist=True, manage_ticks=False,
                            showfliers=True,
                            flierprops=dict(marker="o", markersize=3.5,
                                            markerfacecolor=MCOLOR[m],
                                            markeredgecolor="none", alpha=0.7))
            for box in bp["boxes"]:
                box.set(facecolor=MCOLOR[m], alpha=0.55, edgecolor="black",
                        linewidth=0.8)
            for med in bp["medians"]:
                med.set(color="black", lw=1.3)
            for whisk in bp["whiskers"] + bp["caps"]:
                whisk.set(color="black", lw=0.8)
        ax.axhline(0.5, color="k", lw=0.8, ls="--", alpha=0.5)
        ax.set_xticks(xticks)
        ax.set_xticklabels(N_VALS)
        ax.set_title(f"$p = {p}$")
        ax.set_xlabel("$n$")
    axes[0].set_ylabel("relative Procrustes error")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=MCOLOR[m], alpha=0.55,
                             ec="black", lw=0.8) for m in METHODS]
    fig.legend(handles, [LABEL[m] for m in METHODS],
               loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print("wrote", OUT)

    print("\nMedian Procrustes error:")
    piv = (ok.groupby(["p", "n", "method"])["procrustes"]
           .median().unstack("method")[METHODS])
    print(piv.to_string(float_format="{:.3f}".format))

    print("\nMedian fit time (s):")
    t = (ok.groupby(["p", "n", "method"])["time_sec"]
         .median().unstack("method")[METHODS])
    print(t.to_string(float_format="{:.1f}".format))


if __name__ == "__main__":
    main()
