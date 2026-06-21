# simulation_1 — `zqe_q` (Gaussian-MAP + quantile encoder) findings

*2026-06-21. Question: does the quantile-projected encoder let ZQE win on **gllvm's turf**
(small-n dense Poisson, where plain `zqe` loses to R `gllvm`)?*

## Setup

Added a third arm `zqe_q` to the sweep: the **same** Gaussian-MAP (log1p) encoder as `zqe`, then
each latent margin projected to the prior `N(0,1)` via a rank→`Phi^{-1}` (PIT) transform
(`sweep.QuantileMapEncoder`). Run on all 240 reps (12 settings × 20), appended to the existing
result files without recomputing `zqe`/`gllvm` (method-aware `sweep.run_setting`).

## Directly comparable — verified

- **Identical fit recipe.** Both arms are `ZQEAutoFitter(g, encoder_factory=…, seed=seed, **ZQE_KW)
  .fit(Y)` on the same `fresh_decoder` and `seed`; the **only** difference is `encoder_factory`
  (`MapEncoderGaussianLog1p` vs `QuantileMapEncoder`). Same `ZQE_KW` (the safe recipe), same data.
- **Both converged 100%** of reps (every n). So the safe recipe is not under-serving `zqe_q` — the
  comparison is not confounded by optimization; any difference is the **encoder**, not the fit.

## Result — quantile does NOT win on gllvm's turf

Mean relative orthogonal Procrustes error (lower = better):

| n | gllvm | zqe | zqe_q | zqe_q vs zqe |
|---|---|---|---|---|
| **20** (turf) | 0.40–0.50 | 0.38–0.44 | 0.47–0.57 | **−16…−31% (worse)** |
| 100 | 0.14–0.15 | 0.17–0.18 | 0.17–0.18 | ~0% (wash) |
| 500 | 0.06–0.88* | 0.07–0.08 | 0.07–0.08 | +0…+3% (marginal) |

Overall mean: **`zqe` 0.218 < `zqe_q` 0.250 < `gllvm` 0.300**. Paired: `zqe_q` beats `zqe` only 36%
of reps, beats `gllvm` only 18%.

\*gllvm catastrophically fails at a few large-n settings (p10n500→0.18, p100n500→0.88); where
`zqe_q` "beats" gllvm it is only there, and plain `zqe` ties it — a ZQE *stability* story, not a
quantile win.

## Verdict

- At **small n** (exactly gllvm's turf) the rank→N(0,1) projection over so few points **backfires**
  — it over-imposes the marginal and injects noise.
- At **moderate/large n** it is a wash-to-marginal vs `zqe` (matches the n=2000
  `playground/poisson_quantile.ipynb` finding), consistent with the theory: the projection's
  dominant effect is a **scalar de-shrink that cancels in the centered $m_1-m_2$ equation**, so it
  cannot buy W-efficiency.
- **Decision: do not adopt quantile as a gllvm-turf efficiency lever.** It keeps its value for
  *robustness* (calibrated $\hat\mu$ for Pearson-residual Huberization) and *embeddings* (N(0,1)
  channels), not for loading efficiency. (Also recorded in `paper/CLAUDE.md`.)

Data: `results/q*_rep*.csv`, `method='zqe_q'` (one stray `gllvm` failure flagged, harmless).
