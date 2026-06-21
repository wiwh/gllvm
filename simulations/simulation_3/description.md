# Simulation 3 — Binomial GLLVM, size sweep

The **symmetric twin of simulation 1**: the identical size sweep, but every
response is **Binomial** (logit link, `N = 10` trials) instead of Poisson — a
bounded-count twin of the Poisson (unbounded-count) sweep. Same grid, same metric,
same estimators; only the response family changes.

(Robustness to contamination is *not* studied here — that's simulation 2.)

## Why Binomial(N=10) and not Bernoulli

A single-trial (Bernoulli) GLLVM is **not identifiable at small n**: the loadings
can drive the logits to ±∞ and *perfectly separate* the 0/1 outcomes, so the
likelihood is unbounded. The unregularised ZQE estimate then runs away (Procrustes
≫ 1 — observed at n=20, all p); gllvm survives only because its variational prior
regularises the loadings. This is **separation / non-identifiability**, not a
signal-recovery or efficiency problem.

The clean fix is in the **design**, not a regulariser on the estimator: use
`N > 1` trials. With intermediate true probabilities the counts cannot be
perfectly separated, the likelihood is bounded, and the parameter is identifiable.
Verified at the worst cell (p=50, n=20): Bernoulli (N=1) → Procrustes 2.39
(diverges); N=3 → 0.73; N=5 → 0.43; **N=10 → 0.32** (cleanly identifiable, and
already beating gllvm). We use **N = 10**.

## Sweep (identical to simulation 1)

```
q = 2
p = 10, 20, 50, 100
n = 20, 100, 500          → 12 settings, each repeated H times
```

## What changes vs simulation 1 (Poisson)

- **Family**: `BinomialGLM(total_count=10)`, logit link.
- **Loadings**: `wz_scale = 0.7` so success probabilities stay **nicely mixed** in
  ~(.1, .9) (baseline probs spread across responses, few saturated).
- **Statistic**: `T(y) = 4·(y/N − 0.5)` (centred, scaled proportion ∈ [-2,2])
  instead of `log1p`. Any measurable `T` is valid (score-function identity).
- **Encoder**: the Gaussian-proxy MAP solve on `T(y)` (`sweep.MapEncoderGaussianT`).

## Methods

`zqe` (ZQE, `T = 4(y/N−0.5)`, Gaussian encoder) vs `gllvm` (R `gllvm()`,
`family="binomial"`, `Ntrials=10`, **`link="logit"`**, VA). The `link="logit"` is
essential: gllvm's binomial **default is probit**, which reports loadings on a
different scale (~1.8× = π/√3) than the logit-generated truth — without it the
comparison is unfair (gllvm recovers the right *directions* but a wrong *scale*).
Both fit the same data; loadings scored
against the true `W` after Procrustes rotation. As in simulation 1, the question is
the *relative* standing across `p` and `n`, and the **variance / worst-case**, not
just the median.

## Reproducibility / layout

Same conventions as simulation 1: `seed == rep` drives the true model and data; one
CSV per (setting, rep); resumable; failures flagged. Driver: `sweep.py`.
Experiment: `experiment.ipynb`. Analysis: `analysis.ipynb`. Columns:
`results/DATA_DICTIONARY.md`.
