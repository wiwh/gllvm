# Simulation 2 — robustness to outlier contamination (three branches)

ZQE fits an estimating equation built from a statistic `T(y)` **we choose**, and
the model-expectation centering term `m₂ = E_θ[T·η]` supplies the
Fisher-consistency correction for *any* `T` automatically. So we can make the
loadings estimator **robust by design** — bounded influence in the response —
without losing consistency. The Poisson MLE (`gllvm`) cannot: its influence in
`y` is unbounded, and robustifying a marginal likelihood with the latent
integrated out is the hard problem ZQE sidesteps.

Three branches, all fitting the **same misspecified Poisson model** to the **same**
contaminated data; they differ only in the influence function:

| branch | `T(y)` | influence in `y` |
|--------|--------|------------------|
| `gllvm` | (full Poisson likelihood) | **unbounded** |
| `zqe` | `log1p(y)` | logarithmic (down-weighted) |
| `zqe_huber` | `min(log1p(y), c)` | **bounded** (Huberised) |

`c` is a robust cut in log space, `median + 3·MAD` of `log1p(y)` — a Huber-style
threshold estimated from the (contaminated) data, applied in both decoder and
encoder.

## What we want to see: flat until breakdown

- **Flat in magnitude.** The bounded arm should be *dead flat* as the outlier
  value `M` grows — a capped influence cannot be moved by larger outliers — while
  `gllvm` climbs unboundedly and plain `log1p` climbs logarithmically.
- **Flat until breakdown in fraction.** As the corrupted fraction `eps` grows, the
  bounded arm stays controlled until `eps` is large enough to corrupt the robust
  scale `c` itself (the median/MAD break near `eps ≈ 0.5`) — that is the
  estimator's **breakdown point**.

## Design

Fixed clean model `q=2, p=50, n=200` (where clean ZQE ≈ gllvm). Two dose-response
sweeps sharing the clean point `(eps=0, M=0)`:

| sweep | varies | fixed | reveals |
|-------|--------|-------|---------|
| `eps` | `{.02,.05,.10,.20,.30,.40,.50}` | `M = 1000` | breakdown of the robust arm |
| `M`   | `{10,100,1000,10⁴,10⁵}` | `eps = 0.05` | flatness of the robust arm (bounded influence) |

**Contamination**: replace a fraction `eps` of cells with the gross value `M`
("ε-contamination"). Loadings scored against the clean true `W`.

## Reproducibility / layout

Same conventions as simulation 1: `seed == rep`; one CSV per (condition, rep);
resumable; failures flagged — under heavy contamination `gllvm` **fails to fit**
(non-finite starting values / timeout), which is itself part of the result.
Driver: `sweep.py`. Experiment: `experiment.ipynb`. Analysis: `analysis.ipynb`.
Columns: `results/DATA_DICTIONARY.md`.
