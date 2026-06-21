# `results/` — data dictionary (simulation 2: robustness, three branches)

Produced by `simulation_2/experiment.ipynb` (driver: `simulation_2/sweep.py`).
Same layout/conventions as simulation 1, with two condition columns (`eps`, `M`)
and three methods (`gllvm`, `zqe`, `zqe_huber`).

## Files

One CSV **per (condition, rep)**:

```
q2_p50_n200_e{eps*1000:03d}_M{M:06d}_rep{rep:03d}.csv
e.g.  q2_p50_n200_e050_M001000_rep007.csv   (eps=0.05, M=1000, rep 7)
      q2_p50_n200_e000_M000000_rep000.csv   (clean baseline)
```

A file's existence means that rep is finished (resumable; growing `H` only fits
new reps). Load with `sweep.load_results()`.

## Schema (long / tidy — one row per scalar parameter)

| column | type | meaning |
|--------|------|---------|
| `q`, `p`, `n` | int | fixed clean model size (2, 50, 200) |
| `eps` | float | contamination fraction (0 = clean) |
| `M` | int | outlier value injected into contaminated cells (0 = clean) |
| `rep` | int | replicate, `0 … H-1` |
| `seed` | int | `seed == rep`; drives the true model, the data, **and** the contamination mask |
| `method` | str | `true`, `zqe` (T=log1p), `zqe_huber` (T=min(log1p,c)), or `gllvm` |
| `failed` | float | `1.0` if the fit raised (e.g. gllvm non-finite/timeout under contamination), else `0.0`; `NaN` for `true`. When `1.0`, `value`/`time_sec`/`converged`/`procrustes` are `NaN`. |
| `time_sec` | float | wall-clock fit time; `NaN` for `true`/failed |
| `converged` | float | ZQE convergence flag; `NaN` for `gllvm`/`true` |
| `procrustes` | float | relative orthogonal Procrustes error vs the **clean true** `W`; `NaN` for `true` |
| `param` | str | `W` (loading entry) or `b` (intercept entry) |
| `i` | int | response index `0 … p-1` |
| `j` | int | latent index `0 … q-1` for `W`; `-1` for `b` |
| `value` | float | parameter value (loadings Procrustes-rotated into the true gauge; intercepts as-is) |

`failed`/`time_sec`/`converged`/`procrustes` are per-fit scalars repeated on every
row of a `(rep, method)` block.

## Conditions (two dose-response lines sharing the clean point)

* **clean**: `eps=0, M=0`.
* **`eps` line** (breakdown): `eps ∈ {.02,.05,.10,.20,.30,.40,.50}` at `M = 1000`.
* **`mag` line** (flatness): `M ∈ {10,100,1000,10⁴,10⁵}` at `eps = 0.05`.

Reconstruct in analysis by filtering: `eps`-line `= (M==1000) | (eps==0)`;
`mag`-line `= (eps==0.05) | (eps==0)`. The condition `(eps=0.05, M=1000)` lies on
both (the crossing point).

## Methods (differ only in influence function)

- **`gllvm`** — R `gllvm()` (VA), full Poisson likelihood → unbounded influence in `y`.
- **`zqe`** — ZQE, `T = log1p`, Gaussian-MAP encoder. Logarithmic influence.
- **`zqe_huber`** — ZQE, `T = min(log1p(y), c)` with `c = median + 3·MAD` of
  `log1p(y)` (robust log-space cut), applied in both decoder and a Huberised MAP
  encoder (`sweep.MapEncoderHuber`). **Bounded** influence by design; the `m₂`
  centering term keeps it Fisher-consistent.
