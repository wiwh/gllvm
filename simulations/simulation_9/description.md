# simulation_9 — feasibility raster: ZQE vs R gllvm over (q, p)

**Distinct from simulation_7.** sim_7 is the *ZQE-only* high-p scaling story (q=10 fixed, p up to
50000, procW stays flat — "ZQE doesn't break"). sim_9 is the *comparison* story: a 2-D raster over
latent dim `q` × responses `p` (fixed `n=500`) showing **where each method can run at all** — gllvm
greys out over most of the grid; ZQE fills it.

## What it produces
- `results/raster.csv` — one row per `(method, q, p)`: `seconds_cpu`, `seconds_gpu`, `procW`, `status`.
- `results/loadings/{method}_q{q}_p{p}.npy` — the full estimated `Ŵ` for every cell (truth is
  regenerable: `highdim.make_data(p, q, 500, seed=0)`).
- `feasibility_raster.png` (+ `paper/figures/sim7_feasibility_raster.png`) — 2×2: wall-time and
  Procrustes error, gllvm vs ZQE; grey = gllvm infeasible.

## Design
- Grid: `q ∈ {1,2,3,5,8,10,15,20}` × `p ∈ {100…50000}`, `n=500`, 1 rep (seed=0).
- **Why `q` is capped at 20.** With the row-sparsity cap below, latents past ~20 add no new information
  per response (each response still sees ≤5 latents), so Procrustes recovery would worsen for a reason
  *orthogonal to scaling* — it muddies the "scale" story rather than telling it. `q=20` is already large.
- **Why Procrustes worsens at high `q` (expected, not a failure).** The sparsity cap reduces signal per
  latent: each response is only hit by ~5 of the 20 latents, so the effective information per factor
  shrinks as `q` grows. Nobody expects all 20 latents to affect all 50 000 responses.
- **Row-sparsity cap on the truth (`MAX_LV_PER_RESPONSE=5`).** Each response is affected by at most
  ~5 latents, *independent of `q`* — i.e. each row of the true loading matrix has ≤ ~5 non-zeros.
  This is essential at large `q`: with a dense generation (the naive `responses_per_latent=p//2`),
  each response would be hit by ~`q/2` latents, so `η_j = Σ_k w_jk z_k` grows with `q` and
  `exp(η_j)` (the Poisson rate) saturates/overflows by `q≈20–50`, producing degenerate all-huge
  counts and exploding fits. Capping latents-per-response keeps `η` bounded at every `q`, so the
  raster measures *scaling*, not overflow. Implemented in a **raster-local `make_data`** (sets
  `responses_per_latent ≈ 5p/q`) so sim_7's `highdim.make_data` is untouched.
- ZQE = sim_7's exact *fit* recipe (`highdim.fit`): Poisson-MAP encoder,
  `ZQE_KW` (steps_per_round=150, max_rounds=2, refine_lr=0.5, ema), `l2=0.001/n`.
- **Fair timing**: ZQE timed on **CPU** for `p ≤ 10000` (CPU is faster there, and it's the fair
  CPU-vs-gllvm comparison) and on **GPU** everywhere (GPU only pays off at p ≥ tens of thousands).
  gllvm is CPU (R). Lead the figure with the *feasibility* map (hardware-robust); footnote the time.
- **Time budget (same for ALL three methods): 30-min HARD cutoff + 15-min SOFT budget.** Per `q`, sweep
  `p` increasing. A cell finishing under the SOFT budget (900s) → continue. A cell finishing *between*
  SOFT and the HARD cutoff (900–1800s) → **accepted (ok), but it is the last `p` tried at that `q`** —
  so at most ONE >15-min result per `q`. A cell over the HARD cutoff (1800s) or that OOMs/fails →
  greyed. Either way all larger `p` at that `q` are greyed without running (feasible region is a
  staircase in `p·q`). The soft budget lets the *one* near-the-wall point land in the figure, giving a
  smooth nonlinear blow-up instead of an abrupt wall (otherwise gllvm appears to "fail too fast").
- gllvm: `sd.errors` off (point fit), `maxit=4000`, subprocess `timeout=1800s` (the HARD cutoff).

## Run / resume (restartable any time)
```
python raster_sweep.py     # resumes from results/raster.csv; skips done cells; respects gllvm staircase
python make_raster.py      # renders from partial CSV
```
Per-cell checkpoint (CSV row + `.npy` flushed before next cell) → safe to kill/resume across sessions.
