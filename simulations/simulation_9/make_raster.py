"""
Render the feasibility raster from results/raster.csv: gllvm vs ZQE-CPU vs ZQE-GPU over (q, p), n=500.

2 rows (wall-time, Procrustes error) x 3 columns (gllvm, ZQE-CPU, ZQE-GPU). Cells that timed out /
failed / are infeasible show GREY. Works on partial CSVs, so run any time during the sweep.

Run: python make_raster.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

mpl.rcParams.update({"font.family": "serif", "font.size": 10, "axes.titlesize": 10.5})
HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "results", "raster.csv"))

Q = sorted(df["q"].unique())
P = sorted(df["p"].unique())
qi = {q: i for i, q in enumerate(Q)}
pi = {p: j for j, p in enumerate(P)}
OK = {"ok"}
METHODS = [
    ("gllvm", "gllvm (R, CPU)"),
    ("zqe_cpu", "ZQE (CPU)"),
    ("zqe_gpu", "ZQE (GPU)"),
]


def grid(method, col):
    M = np.full((len(Q), len(P)), np.nan)
    for _, r in df[df["method"] == method].iterrows():
        v = r.get(col, "")
        if r["status"] in OK and pd.notna(v) and v != "" and str(v)[0] != ">":
            M[qi[r["q"]], pi[r["p"]]] = float(v)
    return M


fig, ax = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
t_cmap = mpl.cm.viridis_r.copy()
t_cmap.set_bad("#cfcfcf")  # NaN -> grey (infeasible/slow); dark = long time = bad
e_cmap = mpl.cm.magma_r.copy()
e_cmap.set_bad("#cfcfcf")
tnorm = LogNorm(15, 1200)

for j, (m, label) in enumerate(METHODS):
    im = ax[0, j].imshow(
        grid(m, "seconds"), aspect="auto", origin="lower", cmap=t_cmap, norm=tnorm
    )
    ax[0, j].set_title(f"{label} — wall-time (s)")
    fig.colorbar(im, ax=ax[0, j], shrink=0.85, label="s")
    im = ax[1, j].imshow(
        grid(m, "procW"),
        aspect="auto",
        origin="lower",
        cmap=e_cmap,
        vmin=0.1,
        vmax=1.0,
    )
    ax[1, j].set_title(f"{label} — Procrustes error")
    fig.colorbar(im, ax=ax[1, j], shrink=0.85, label="proc")

for a in ax.ravel():
    a.set_xticks(range(len(P)))
    a.set_xticklabels(P, rotation=45, fontsize=8)
    a.set_yticks(range(len(Q)))
    a.set_yticklabels(Q, fontsize=8)
    a.set_xlabel("p (responses)")
    a.set_ylabel("q (latent dim)")


def n_ok(m):
    return int((df[df.method == m].status == "ok").sum())


fig.suptitle(
    f"Where can each method run?  Poisson GLLVM, n=500   " f"(cutoff 30 min)",
    fontsize=12,
)
fig.savefig(os.path.join(HERE, "feasibility_raster.png"), dpi=160)
papfig = os.path.abspath(os.path.join(HERE, "..", "..", "paper", "figures"))
os.makedirs(papfig, exist_ok=True)
fig.savefig(os.path.join(papfig, "sim9_feasibility_raster.png"), dpi=160)
fig.savefig(os.path.join(papfig, "sim9_feasibility_raster.pdf"))
print(
    "saved -> simulation_9/feasibility_raster.png  and  paper/figures/sim9_feasibility_raster.{png,pdf}"
)
for m, label in METHODS:
    print(f"  {label:16s} solved {n_ok(m):2d}/{len(Q)*len(P)}")
