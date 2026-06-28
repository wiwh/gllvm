import nbformat as nbf
nb = nbf.v4.new_notebook(); C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def co(s): C.append(nbf.v4.new_code_cell(s))

md(r"""# GLLVM scaling to very high response dimension (no competitor)

Sparse Poisson GLLVM fit by $Z_q$ with **response sub-sampling**: each step uses a random
`FEATURE_BATCH` of the $p$ responses for the encoder, the statistic, and the model fantasies,
so per-step cost is independent of $p$. R `gllvm` ($O(p^2q^2)$ Hessian) and VA/Laplace cannot run
in this regime at all.

**Tune the knobs in the Config cell.** Notes from the build:
- True model is **super-sparse** (each response loads on $0$–`MAX_ACTIVE` of the $q$ latents); with
  dense loadings at large $q$, $\eta=Wz+b$ explodes and $e^\eta\to$ NaN.
- Use a **small learning rate** (`LR` $\approx 0.03$); larger LR ($\ge 0.1$) makes $\hat W$ blow up.
- Recovery (procW) is limited by $n$ and by passes-per-response: `TARGET_PASSES` $\propto$ the number
  of times each response is visited. Good recovery is **linear in $p$** (optimal — $W$ has $pq$
  entries); the *per-step* cost is what's flat in $p$. Larger `FEATURE_BATCH` improves recovery per
  pass at higher per-step cost.""")

co(r"""import sys, os, time
sys.path.insert(0, os.path.abspath("../../src"))
sys.path.insert(0, os.path.dirname(os.path.abspath("__file__")) if "__file__" in globals() else ".")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
import highdim
from gllvm import diagnostics
from gllvm.autofit import orthogonal_align, procrustes_error""")

md(r"""## Config — tune here
Start small; scale `P_GRID` up to `[..., 50000, 100000]` once the knobs look good
(runtime grows ~linearly in $p$ and in `TARGET_PASSES`).""")

co(r"""N            = 500                       # observations
Q_LIST       = [20, 50]                   # latent dimensions to sweep
P_GRID       = [1000, 5000, 10000]        # response dims (scale up to 50_000, 100_000)
FEATURE_BATCH = 1000                      # responses processed per step (tunable; try 2000)
TARGET_PASSES = 120                       # ~times each response is visited (recovery vs cost)
LR           = 0.03                       # keep small; >=0.1 blows W up
L2           = 0.0                        # optional ridge on loadings: loss += L2*||W||^2 (0 = off)
MAX_ACTIVE   = 5                          # each true response loads on 0..MAX_ACTIVE latents
SEED         = 0""")

md(r"""## Diagnose a single fit
Fit one $(p, q)$ and inspect convergence — use this to tune `LR` / `TARGET_PASSES` /
`FEATURE_BATCH` before launching the full sweep.""")

co(r"""p_diag, q_diag = 10000, 20
Yd, Wt, _ = highdim.make_true(N, p_diag, q_diag, max_active=MAX_ACTIVE, seed=SEED)
ft, dt, pw = highdim.fit(Yd, q_diag, feature_batch=FEATURE_BATCH, target_passes=TARGET_PASSES,
                         refine_lr=LR, warmup_lr=LR, l2=L2, seed=SEED, W_true=Wt)
print(f"p={p_diag} q={q_diag}: fit {dt:.1f}s  procW={pw:.3f}  ||W_true||={Wt.norm():.1f}  ||W_hat||={ft.model.wz.norm():.1f}")
print(f"converged={ft.converged_}  rounds={ft.n_rounds_used_}  restart-change={ft.change_:.4f}")""")

co(r"""# convergence diagnostics (scalar histories — valid under feature sub-sampling)
fig, ax = plt.subplots(2, 2, figsize=(13, 8))
for fn, a, ttl in [(diagnostics.plot_objective, ax[0,0], "objective -(m1-m2): must fluctuate about 0"),
                   (diagnostics.plot_gradnorm,  ax[0,1], "grad norm"),
                   (diagnostics.plot_lr,        ax[1,0], "learning-rate schedule"),
                   (diagnostics.plot_convergence,ax[1,1], "restart change / |avg grad| vs tol")]:
    try:
        fn(ft, ax=a); a.set_title(ttl)
    except Exception as e:
        a.set_title(f"{ttl}\n(unavailable: {type(e).__name__})")
fig.tight_layout(); plt.show()""")

co(r"""# recovery scatter: estimated vs true loadings (Procrustes-aligned)
West = ft.model.wz.detach().cpu(); R = orthogonal_align(Wt, West); Wa = (West @ R).numpy()
Wtn = Wt.numpy(); idx = np.random.default_rng(0).choice(Wtn.size, size=min(20000, Wtn.size), replace=False)
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(Wtn.ravel()[idx], Wa.ravel()[idx], s=3, alpha=.2)
lim = np.abs(Wtn).max(); ax.plot([-lim, lim], [-lim, lim], "k--", lw=.8)
ax.set_xlabel("true loading"); ax.set_ylabel("estimated (aligned)")
ax.set_title(f"loading recovery  (procW={pw:.3f})"); fig.tight_layout(); plt.show()""")

md(r"""## Scaling sweep
Fit time and recovery across `P_GRID` for each $q$. **The fit happens here.**""")

co(r"""dfs = []
for q in Q_LIST:
    dfs.append(highdim.run_sweep(P_GRID, q=q, n=N, feature_batch=FEATURE_BATCH,
                                 target_passes=TARGET_PASSES, refine_lr=LR, warmup_lr=LR, l2=L2,
                                 max_active=MAX_ACTIVE, seed=SEED, device="cpu", verbose=True))
df = pd.concat(dfs, ignore_index=True)
df.to_csv("results_highdim.csv", index=False)
df""")

co(r"""fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for q, mk in zip(Q_LIST, ["o-", "s-", "^-", "d-"]):
    d = df[df.q == q]
    ax[0].plot(d.p, d.fit_s, mk, lw=2, label=f"$q={q}$")
    ax[1].plot(d.p, d.procW, mk, lw=2, label=f"$q={q}$")
ax[0].set_xscale("log"); ax[0].set_xlabel("responses $p$"); ax[0].set_ylabel("wall-clock fit time (s)")
ax[0].set_title("fit time vs $p$"); ax[0].legend(); ax[0].grid(alpha=.3, which="both")
ax[1].set_xscale("log"); ax[1].set_xlabel("responses $p$"); ax[1].set_ylabel("Procrustes error of $W$")
ax[1].axhline(1.0, color="grey", ls=":", lw=1)
ax[1].set_title("loading recovery vs $p$"); ax[1].legend(); ax[1].grid(alpha=.3, which="both")
fig.suptitle(rf"Sparse Poisson GLLVM, $n={'{'}N{'}'}$, feature\_batch$={'{'}FEATURE_BATCH{'}'}$, passes$\approx{'{'}TARGET_PASSES{'}'}$ — R \texttt{{gllvm}}/VA cannot run here", y=1.02)
fig.tight_layout()
fig.savefig("../../paper/figures/highdim_scaling.png", dpi=130, bbox_inches="tight")
plt.show(); print("saved paper/figures/highdim_scaling.png")""")

md(r"""## Notes for tuning
- **`LR`**: if `||W_hat||` $\gg$ `||W_true||` or procW $>1$, lower `LR` (try 0.02–0.03).
- **`TARGET_PASSES`**: raise for better recovery (procW down) at linear cost in time.
- **`FEATURE_BATCH`**: larger ⇒ better $\hat z$ per step ⇒ better recovery per pass, higher per-step cost.
- **`N`**: recovery is ultimately $n$-limited; raise `N` for a lower procW floor at large $q$.
- **`L2`**: ridge on the loadings ($\text{loss} \mathrel{+}= L2\cdot\|W\|_F^2$); stabilises the
  overcomplete / large-$q$ regime and shrinks $\|\hat W\|$. Trades a little bias for variance, so
  the $Z_q$ root is no longer exactly centred — keep it small. Default $0$.
- The headline is that the estimator **fits at $p$ where no likelihood method can** at a per-step cost
  independent of $p$; recovering all $pq$ loadings to a target accuracy is linear in $p$ (optimal).""")

nb["cells"] = C
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
out = "/home/willwhite/GitHub/gllvm/simulations/simulation_7/scaling.ipynb"
nbf.write(nb, out); print("wrote", out)
