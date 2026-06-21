# `results/` — data dictionary (simulation 3: Binomial size sweep)

Produced by `simulation_3/experiment.ipynb` (driver: `simulation_3/sweep.py`).
**Identical schema to simulation 1** (the Poisson twin); the only differences are
the response family — **Binomial(N=10)** (logit link) — and the two estimators
compared. (Binomial N>1 rather than Bernoulli N=1 because the single-trial setting
is non-identifiable at small n — perfect separation; see `../description.md`.)

## Files

One CSV **per (setting, rep)**:

```
q{q}_p{p}_n{n}_rep{rep:03d}.csv      e.g.  q2_p50_n100_rep007.csv
```

A file's existence means that rep is finished (resumable; growing `H` only fits
new reps). Load with `sweep.load_results()`.

## Schema (long / tidy — one row per scalar parameter)

| column | type | meaning |
|--------|------|---------|
| `q`, `p`, `n` | int | setting (q=2; p∈{10,20,50,100}; n∈{20,100,500}) |
| `rep` | int | replicate, `0 … H-1` |
| `seed` | int | `seed == rep`; drives the true model and the data |
| `method` | str | `true`, `zqe` (T=4(y/N−0.5)), or `gllvm` (family="binomial", Ntrials=10) |
| `failed` | float | `1.0` if the fit raised, else `0.0`; `NaN` for `true`. When `1.0`, `value`/`time_sec`/`converged`/`procrustes` are `NaN`. |
| `time_sec` | float | wall-clock fit time; `NaN` for `true`/failed |
| `converged` | float | ZQE convergence flag; `NaN` for `gllvm`/`true` |
| `procrustes` | float | relative orthogonal Procrustes error vs the true `W`; `NaN` for `true` |
| `param` | str | `W` (loading entry) or `b` (intercept entry) |
| `i` | int | response index `0 … p-1` |
| `j` | int | latent index `0 … q-1` for `W`; `-1` for `b` |
| `value` | float | parameter value (loadings Procrustes-rotated into the true gauge; intercepts on the logit scale, as-is) |

`failed`/`time_sec`/`converged`/`procrustes` are per-fit scalars repeated on every
row of a `(rep, method)` block.

## Methods

Both fit the same Bernoulli data; loadings differ only by estimator:

- **`zqe`** — `ZQEAutoFitter`, `BinomialGLM(total_count=10)` decoder, statistic
  `T(y) = 4·(y/N − 0.5)`, Gaussian-proxy MAP encoder on the same `T`
  (`sweep.MapEncoderGaussianT`).
- **`gllvm`** — R `gllvm()` (VA), `family="binomial"`, `Ntrials=10`, `link="logit"`
  (logit required to match the logit-generated truth; gllvm's binomial default is
  probit, which differs by the ~1.8× = π/√3 logit/probit scale factor).
