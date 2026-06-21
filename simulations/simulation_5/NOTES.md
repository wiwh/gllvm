# GP-GLLVM — development notes & open problems (updated 2026-06-21)

Context dump for the GP-GLLVM line of work (ZQE estimating a Gaussian-process latent
GLLVM). Prototype lives in `playground/gp-gllvm/` (`gp_gllvm.py`, `gp_fit.py`, demo +
fit notebooks); the scalable/main-paper version is here in `simulation_5/`
(`gp.py`, `scaling.ipynb`, `description.md`).

## RESOLUTION (2026-06-21) — read this first

The earlier "OPEN PROBLEMS / wrong equation" sections below are **superseded**. After a literature
review (`gp_factor_analysis_literature.md`) and ~15 fresh-data experiments, the conclusion is:

- **Cross-latent correlation B is a non-identification, not a bug.** ALL standard GP factor
  analysis (GPFA, Yu 2009, + variants) assumes **independent latents (B=I)** — the cross-latent
  correlation is a gauge absorbed into the loadings (`W↔chol(B)`; lower-tri W removes O(q) rotation
  but NOT this gauge). We reproduced it: shared kernel ⇒ ρ pure gauge; distinct kernels ⇒ ρ
  identified but encoder-self-consistency-biased (≈0.9, worse with distinctness). See
  `playground/gp-gllvm/gp_gllvm_bias.ipynb`.
- **DECISION: assume B = I, estimate per-latent timescales + loadings** (the identified GPFA
  estimand). With B=I the estimator is **clean**: 10 cold-start fresh-data runs give
  ℓ̂ unbiased to ~2% (e.g. true [1,3] → [1.01±0.03, 3.02±0.08]), procW ≈ 0.027 ± 0.008, **zero
  divergence**. See `playground/gp-gllvm/gp_gllvm_timescales.ipynb` (convergence diagnostics + the
  10-seed table). This is the paper backbone.
- **ε is recoverable from the full inverse `y→ẑ→ε̂` only when ℓ ≈ Δt** (compatible sampling);
  oversampled ℓ≫Δt ⇒ Σ collapses, ε unrecoverable (correct Bayesian behaviour, not a flaw). See
  `gp_gllvm_roundtrip.ipynb`. (The earlier multi-output divergence was largely this oversampling.)
- **Always validate ≥10× from FRESH data** (new W₀ + data each seed). Single-seed runs misled us
  repeatedly this session — the η-route looked great on one seed, diverged ~60% across fresh data.
- **The centering `m₂(θ)` cannot be frozen/EMA-tracked** as a parameter: it is encoder-coupled to ℓ
  (computed from imputed ẑ), so freezing removes the restoring force and ℓ→∞ even from the truth.
  A deterministic version needs an **analytic** `m₂(θ)` recomputed each step (not frozen). See
  `gp_gllvm_centering_param.ipynb` (negative result).

- **Efficiency (anchor result, to develop later):** with the **canonical statistic** the centered
  ZQE equations are *exactly the MLE (score) equations* — so for Gaussian data the estimator
  **coincides with the MLE, no efficiency loss** (analytic, not just empirical). The centering
  `m₂(θ)=E_θ[T(Y)·η]` is then **closed-form** (Gaussian-MGF expectation; extends to e.g. Poisson
  with T=y) ⇒ a fully deterministic, no-sampling fitter. Caveat: canonical T=y is what makes both
  the MLE-equivalence and the closed-form centering work, but T=y can be numerically unstable at
  random init (why we used T=log1p) — reconcile (clamp / warm start) before leaning on it.
  Framing: *one scalable, likelihood-free recipe that equals the MLE where the MLE exists and
  extends to where it doesn't (non-Gaussian, huge p, n→millions).*

**Paper status:** core estimator + result solid and identified. Still TODO for a paper: promote the
B=I per-latent estimator into `gp.py` (currently single shared ℓ), confirm it **scales** like the
single-ℓ result (flat time vs n), add a **GPFA baseline** (`elephant`) comparison, sweep more
sim conditions (true ℓ, q>2, count level), and ideally one real-data example.

## Model
Grouped/longitudinal GP-GLLVM. `B` groups, each `G` observations on a grid `t`
(time/space). Within a group each latent factor is a GP; groups independent; Poisson:
```
per group:  ε ~ N(0, I_{G×q}),  z[:,k] = L(ℓ_k) ε[:,k]  (L = chol K(t)),  y ~ Poisson(e^{Wz+b}).
```
Data generated **once** with full within-group correlation.

## Estimator (composite ZQE) — the design that works
- **Latent is the white ε; kernel params live in the DECODER** (`ε → z=L(ℓ)ε → η=Wz+b`),
  so `ℓ` is in `η` and is learned by **plain backprop**.
- **Encoder = no-grad imputation**: per-obs Gaussian-MAP → `Ẑ`, then whiten
  `ε̂ = L(ℓ)⁻¹ Ẑ`. Because `ε̂` is detached, the `L(ℓ)` reused in the decoder
  (`η = W·L(ℓ)·ε̂ + b`) does NOT cancel → `ℓ` gets a real gradient. (Same trick as the
  GLLVM handles `W,b`.)
- **K-subset / GP marginal theorem (the scaling key):** any subset of a group's `G`
  points is itself an exact GP draw `N(0, K(t_subset))`. So fitting samples a random
  **K-subset** per group per step → every op is `K×K`, never `G×G`. **`K` is a tunable
  knob** (cost vs efficiency). Fantasies are also K-subsets.
- Per-step cost `O(batch·K·p + K³)` — **independent of `G` and `B`**.
- (Block subsetting for GPs = Vecchia / composite likelihood; novelty = the
  *likelihood-free* realization for non-Gaussian GLLVMs, where the block likelihood
  integrates an intractable non-Gaussian latent.)

## Validated results (single shared lengthscale, q=2, p=30, RBF, true ℓ=2.0; GPU)
- **Recovery:** est ℓ ≈ 1.9–2.0, loadings Procrustes ≈ 0.02–0.05. Lengthscale recovery
  holds across true ℓ ∈ {1,2,3} → {1.02, 1.96, 3.14} (single-ℓ playground fit).
- **Scaling — fit time FLAT:**
  - group size `G` ∈ {50,200,1000} (n=B·G up to 2M), B=2000, K=15: fit ≈ 3.5 s for all,
    recovery improves with G (Procrustes 0.047→0.015).
  - #groups `B` ∈ {1k,5k,20k} (n up to 4M), G=200, K=15: fit ≈ 3.3 s for all.
  - `K` knob at G=200: K=5→0.030/2.6s, K=15→0.015/3.1s, K=40→0.023/4.0s.
- Full-group GP likelihood is `O(G³)` and non-Gaussian-intractable; `gllvm` has no GP
  latent. So this is uncontested at scale.

## Identifiability — the crux (corrected understanding)
Two DIFFERENT "latent correlations"; do not conflate (I did, initially):

1. **Instantaneous within-observation cross-latent covariance** `Σ_z = Cov(z_i)` (q×q at
   a single point). **NOT separately identifiable in the exploratory model** — it is
   absorbed into the loadings: `z~N(0,Σ_z)` with `W` ≡ `z'~N(0,I)` with `W'=W·chol Σ_z`
   (same likelihood). It is a gauge/convention (set `Σ_z=I`). Becomes a real estimand
   only with **confirmatory/constrained loadings** (fixed zero pattern / scale, à la
   CFA/SEM), where the rotation gauge is removed by construction.

2. **Across-time/space GP correlation** (the lengthscales `ℓ_k`, the kernel structure).
   **IS identifiable** — and this is the object of interest. The loadings act on a
   *single* `z_i` (one observation) and are fixed across observations, so they CANNOT
   encode how observations correlate over time/space. That correlation lives in the GP
   prior, in a different "place" than the loadings. (This is the point the loadings
   argument does NOT apply to — my earlier "it's in the loadings" was about object #1,
   not this.)

So: per-latent lengthscales are a genuine, identifiable estimand (NOT absorbed into
loadings), and per-latent *different* lengthscales are a free design choice, not a cost.

## OPEN PROBLEM: per-latent lengthscales don't separate (q>1)
Tried q=2 with **distinct** true lengthscales ℓ=[1.0, 4.0] (each latent its own kernel),
fit with per-latent `log_ell` (init [1.5,2.5]). Result:
```
TRUE ell (sorted) = [1.0, 4.0]
EST  ell (sorted) = [1.875, 1.909]      <-- collapsed to ~1.9, did NOT separate
loadings procrustes = 0.027             <-- loadings fine
```
**The lengthscales collapsed to a common value; the distinct per-latent kernels were
not recovered**, even though loadings were.

**Hypothesis (why):** the per-obs MAP encoder returns `Ẑ` in a **rotation-ambiguous
basis** (set by the current `W`), so its columns are *mixtures* of the true latent
factors. Whitening column `k` with `L(ℓ_k)` then applies the wrong kernel to a mixed
column → the per-latent kernels average out → both `ℓ_k` drift to the mean (~1.9). The
across-time identifiability is real *in principle* (distinct kernels break the rotation
gauge), but the current ZQE construction does not **exploit** it: nothing ties kernel
`k` to a specific true latent direction.

**Directions to try (next):**
- Tie kernels to loading directions: jointly estimate a rotation so column `k` of the
  whitened latent is the factor with kernel `ℓ_k` (break the basis ambiguity inside the
  estimator). E.g. estimate `W` with an identifying constraint (lower-triangular /
  varimax / fixed) so latent columns are pinned, *then* per-latent kernels separate.
- A cross-observation (lagged) statistic in the estimating equation that is sensitive
  to per-factor temporal decay in the *identified* basis (the column-wise lag
  autocorrelation of `V=WA` distinguishes kernels — match that).
- Confirmatory loadings (mask) — removes the rotation gauge → per-latent kernels (and
  the factor correlation matrix) should both become recoverable.
- Sanity: does the *likelihood* (small n, full GP) separate [1,4]? If yes, it's purely
  an estimator-construction issue, not identifiability.

## Files
- `playground/gp-gllvm/gp_gllvm.py` — model + block encoder (MAP→whiten).
- `playground/gp-gllvm/gp_fit.py` — composite ZQE fitter (single ℓ).
- `playground/gp-gllvm/gp_gllvm_demo.ipynb` — correlation = kernel; whitening de-correlates.
- `playground/gp-gllvm/gp_gllvm_fit.ipynb` — single-ℓ recovery.
- `simulation_5/gp.py` — self-contained scalable model + `fit_gp_zqe(subset_K=…)`.
- `simulation_5/scaling.ipynb` — scaling (flat fit time vs G, B), K-knob, ℓ-recovery.
- `simulation_5/description.md` — main-paper framing.
- per-latent-ℓ probe: `/tmp/gp_perlatent_probe.py` (collapse result above) — to be moved
  in once the construction is fixed.

## Encoder architecture (corrected, 2026-06-18, per GB)
NEVER a per-obs encoder over the whole dataset. ALWAYS K-blocks. Within a K-block:
(1) per-obs MAP **independently** (conditionally independent given z) → Ẑ (K×q);
(2) **then JOINTLY whiten the whole block** with the GP covariance `L_Σ⁻¹` to get the
K(×q) ε's. The joint whitening is where the temporal kernels AND the cross-latent B
live — NOT in the MAP, and NOT per-column. Decoder re-correlates with `L_Σ`.

## Identifiability of the multivariate (cross-latent) correlation — RESOLVED (it IS identifiable)
Distinguish: instantaneous within-obs cross-latent cov (absorbed into loadings, gauge)
vs the **across-time** cross-latent correlation (the GP). The latter IS identifiable:
for q=2, `Cov(η_i,η_j) = w₁w₁ᵀK₁ + w₂w₂ᵀK₂ + ρ(w₁w₂ᵀ+w₂w₁ᵀ)K×`. With DISTINCT temporal
shapes {K₁,K₂,K×}, the diagonal components fix w₁,w₂ and the cross component fixes ρ.
Loadings are per-single-obs/fixed → cannot absorb ρ (it rides a distinct cross-temporal
signature). So per-factor ℓ_k (temporal) and B (multivariate) are SEPARATE, both
identifiable. [Earlier "B absorbed into loadings" applied only to the LMC/shared-kernel
case; the direct construction with a distinct cross-kernel is identifiable.]

Valid PSD cross-kernel (Gaussian cross-spectral): `K×(d)=ρ·√(2ℓ_kℓ_l/(ℓ_k²+ℓ_l²))·
exp(−d²/(ℓ_k²+ℓ_l²))` (arithmetic mean of ℓ², sub-unit amplitude). The geometric-mean-of-
KERNELS form is NOT PSD (fails at high frequency).

## OPEN PROBLEMS (multivariate / per-latent) — STILL UNSOLVED, two failed attempts
1. **Per-column whiten (B=I), per-latent ℓ:** distinct ℓ=[1,4] COLLAPSE to ~[1.9,1.9]
   from random W (loadings ok). ℓ-warm-up did NOT fix it.
2. **Joint K·q whiten with B (correct architecture + PSD cross-kernel):** DIVERGED —
   ℓ=[1.8,1.9] (collapsed), ρ→−0.95 (wrong sign, boundary), procrustes(W)=1.8 (worse
   each step). Diagnosis: the encoder whiten/decoder re-correlate **cancel in value**
   (`L_Σ L_Σ⁻¹ Ẑ = Ẑ`), so ∂m₁/∂(ℓ,B) comes only from autodiff through the cancelled
   `L_Σ`; that works for a single scalar ℓ but is mis-signed/ill-conditioned for the
   cross-covariance B → ρ to boundary → W diverges.
   **→ Needs the estimating equation for (ℓ_k, B) DERIVED explicitly, not obtained by
   autodiff-through-cancellation. Stop trial-and-error here.**
3. **The "root" is ENCODER-DEPENDENT → the (ℓ,B) equation is provably not consistent.**
   (2026-06-18, `playground/gp-gllvm/gp_gllvm_multioutput.ipynb` — full per-iterate logging,
   2×2 grid {Gaussian-MAP, Poisson-Newton-MAP} × {truth-init, cold-init}.)

   | MAP | truth-init | cold-init |
   |-----|-----------|-----------|
   | Gaussian-on-log1p | **stable**, ρ≈+0.42 (biased low), procW≈0.05 | diverges, ρ→−0.96, procW→2.1 |
   | Poisson (Newton)  | **DIVERGES**, ρ→−0.98, procW→2.3 | hits truth (ρ≈+0.46, procW≈0.02 @~step 800) then drifts to ρ→−0.95 |

   - Swapping ONLY the (detached, no-grad) encoder MOVES the root: Gaussian-MAP makes truth a
     (biased) stable root; the *real* Poisson-MAP does NOT — start AT the truth and (ℓ,B) walk to
     the ρ=−1 boundary. Cold-Poisson even passes *through* the truth and leaves it. A consistent
     Z-equation's root is encoder-invariant (the score identity for W,b). So this is **wrong
     equation**, not bad optimisation / basin (earlier "basin" framing retracted).
   - **Mechanism, sharpened:** the cancellation makes η's VALUE independent of (ℓ,B)
     (`z_dec=L_ΣL_Σ⁻¹Ẑ=Ẑ`, and Ẑ=per-obs MAP with N(0,I) prior uses only W,b). So the ZQE
     statistic T(y)·η — which is the score for (W,b) — carries NO information about (ℓ,B) in its
     value; its (ℓ,B)-gradient is a spurious moment that depends on Ẑ (hence on the MAP) and has
     no root at the truth. The loss stays ≈0 throughout the divergence (failure invisible in the
     objective). Single-ℓ only *appeared* to work (one spurious scalar pointed uphill); B exposes it.
   - **Fix direction (unchanged, now the only path): a statistic whose VALUE depends on (ℓ,B)** —
     a cross-observation / lag moment, e.g. match imputed-latent lag-covariance E[Ẑ_i Ẑ_jᵀ] (or
     E[T(y_i)T(y_j)ᵀ] over within-block pairs) to the model's L_Σ-implied covariance. Per-obs
     T(y)·η can never identify (ℓ,B). ρ-clipping / slow B-freeing only delay the slide.

## Status line (2026-06-21)
Single-ℓ GP-GLLVM at scale: **works** (kernel + loadings, n→millions, flat time) — solid.
**Per-latent ℓ_k with B=I: works cleanly** — unbiased timescales (~2%), procW≈0.03, stable over 10
fresh-data cold starts (`gp_gllvm_timescales.ipynb`). This is the estimator.
Cross-latent **B: dropped** — non-identified gauge (lit-confirmed; `gp_factor_analysis_literature.md`).
Next: scale the per-latent estimator in `gp.py`, add a GPFA baseline, broaden sims.
(Earlier "estimator provably wrong" framing retracted — it was the cross-latent-B gauge + an
oversampled regime, not a flaw in the per-latent estimator.)
