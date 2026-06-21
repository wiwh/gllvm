# `results/` — data dictionary

Produced by `simulation_1/experiment.ipynb` (driver: `simulation_1/sweep.py`).

## Files

One CSV **per (setting, rep)**:

```
q{q}_p{p}_n{n}_rep{rep:03d}.csv      e.g.  q2_p50_n100_rep007.csv
```

A file's existence means that rep is finished — this is the whole resumability
story. Growing the number of reps (e.g. `H = 20 → 100`) only fits and writes the
*new* rep files; existing ones are never recomputed. Load everything with
`sweep.load_results()`.

## Schema (long / tidy — one row per scalar parameter)

| column | type | meaning |
|--------|------|---------|
| `q` | int | latent dimension (2 throughout this sweep) |
| `p` | int | number of responses |
| `n` | int | number of observations |
| `rep` | int | replicate index, `0 … H-1` |
| `seed` | int | RNG seed for this rep. **`seed == rep`**, and it drives *both* the true model and the simulated dataset, so each rep is fully reproducible from its seed alone. |
| `method` | str | `true` (the data-generating parameters), `zqe`, `zqe_q`, or `gllvm` |
| `failed` | float | `1.0` if this method's fit raised an error (R failure, NaN blow-up, …), else `0.0`; `NaN` for `true`. When `1.0`, all of `value`/`time_sec`/`converged`/`procrustes` for the block are `NaN`. A failed fit is still recorded so the rep is complete and never silently retried; the other method and the `true` block are unaffected. |
| `time_sec` | float | wall-clock fit time for this method; `NaN` for `true` and for failed fits |
| `converged` | float | ZQE convergence flag (`1.0`/`0.0`); `NaN` for `gllvm` (no such flag) and `true` |
| `procrustes` | float | relative orthogonal Procrustes error of this method's loadings vs `true`; `NaN` for `true` |
| `param` | str | `W` (loading matrix entry) or `b` (intercept entry) |
| `i` | int | response index, `0 … p-1` |
| `j` | int | latent index `0 … q-1` for `param='W'`; `-1` for `param='b'` |
| `value` | float | the parameter value (see rotation note below) |

`time_sec`, `converged`, `procrustes` are **per-fit scalars repeated** on every
row of that `(rep, method)` block.

## How parameters are stored

* **Loadings (`param='W'`)** are stored **after Procrustes rotation into the true
  gauge**: for each estimate we find `R* = argmin_R ||W_true − W_est R||_F` over
  the orthogonal group (reflections allowed) and store `W_est @ R*`. This removes
  the `O(q)` rotational ambiguity so estimates are directly comparable, element by
  element, to `true`. (The `true` rows store the raw true loadings.)
* **Intercepts (`param='b'`)** are stored **as-is** — the latent rotation acts on
  `z` only (`z ~ N(0, I)`), so the per-response intercept `b` is invariant to it
  and needs no transformation. For `gllvm` these are `fit$params$beta0`.

## Reconstructing matrices

```python
import sweep
df  = sweep.load_results()
one = df.query("p==50 and n==100 and rep==0 and method=='zqe'")
W   = one.query("param=='W'").pivot(index='i', columns='j', values='value').to_numpy()  # (p, q)
b   = one.query("param=='b'").sort_values('i')['value'].to_numpy()                       # (p,)
```

## Estimators

* **`zqe`** — `gllvm.autofit.ZQEAutoFitter`, Gaussian-MAP encoder
  (`MapEncoderGaussianLog1p`), statistic `T = log1p` on decoder and encoder.
  Knobs (`sweep.ZQE_KW`) match `simulations/poisson.ipynb`.
* **`zqe_q`** — identical to `zqe` but with the **quantile-projected** Gaussian-MAP
  encoder (`sweep.QuantileMapEncoder`): the MAP latent is marginally projected to
  the prior `N(0,1)` per dimension via a rank→`Phi^{-1}` (PIT) transform. Same MAP,
  same `T=log1p`; the only difference is the calibration step.
* **`gllvm`** — R's `gllvm::gllvm()` via `gllvm.r_gllvm.RGllvm` (`method="VA"`,
  `family="poisson"`).

All estimators are fit to the **same** simulated dataset for each rep. The result
files are **method-aware**: `zqe_q` was added to already-finished reps without
recomputing `zqe`/`gllvm` (see `sweep.run_setting`).
