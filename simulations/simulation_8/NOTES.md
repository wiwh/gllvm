# simulation_8 — standard errors / confidence intervals for ZQE loadings

Goal: give the ZQE loading estimates **calibrated standard errors** (so the paper's "inference"
claim is real, not just point estimation), and validate them by **coverage**.

## Method (the validated one)

`varboot.param_bootstrap_resolve` — a **common-random-numbers parametric bootstrap** of the
*actual* estimator:

1. fit `theta_hat` (ZQEAutoFitter, Poisson-MAP encoder, `T=log1p`, ridge `l2=c/n`);
2. for each replicate `h`: simulate `Ytil ~ f_{theta_hat}`, then **re-solve the same estimator**
   warm-started at `theta_hat` (`warmup_max_epochs=0`) with a **fixed seed** (CRN);
3. the spread of the replicate estimates is `Var(theta_hat)`.

CRN (fantasy held fixed across replicates) makes the bootstrap spread reflect *data* sampling
variability, not Monte-Carlo noise in the centering. SEs are read on the rotation-invariant Gram
functionals `(WW^T)_jk` (no gauge bookkeeping).

## The key finding (cost us a long debugging arc)

**The bootstrap must use the estimator's own statistic.** The estimator uses `T=log1p`
(deliberately less efficient than canonical). A cheap bootstrap that refits a **canonical Poisson
GLM (`T=Y`)** measures the *more-efficient* variance → SE too small → **CIs too narrow →
undercoverage** (SE ratio ≈ 0.91, coverage ≈ 0.91). Re-solving the actual `T=log1p` estimator
fixes it: **SE ratio → ~1.0, coverage → nominal.** It was the statistic mismatch — not skew, not
the encoder.

## Dead ends (don't repeat)

- frozen-encoder + canonical Poisson-GLM refit (`T=Y`): wrong statistic → SE too small.
- least-squares of `log1p(Y)` on `z`: not the Poisson estimating equation.
- bias-corrected GLM fixed point `theta<-theta_d-(theta_q-theta)`: not contractive (drifts) and
  it's the `T=Y` root.
- warm-started-but-not-converged re-solve: warm-start compresses the spread (ratio 0.82).
- a one-shot deterministic solve with the centering *fixed*: structurally impossible — θ is
  identified only through the centering's θ-dependence, so fixing it removes identification. (A
  deterministic L-BFGS / sandwich version is possible but left for future work.)

## Files

- `varboot.py` — `make_truth`/`sample_data`, `fit_point`, `param_bootstrap_resolve` (the method),
  `param_bootstrap`/`param_bootstrap_crn` (labeled `T=Y` contrasts), `gram_entries`,
  `coverage_experiment*`.
- `run_validation.py` — runs the D-dataset comparison (naive `T=Y` vs matched `T=log1p`), writes
  `results/validation.npz`.
- `make_figure.py` — `results/validation.npz` → `coverage_validation.png` (+ `paper/figures/
  sim8_coverage_validation.png`): (A) SE calibration scatter, (B) CI coverage.

See memory `zqe-bootstrap-se-statistic-match` for the condensed version.
