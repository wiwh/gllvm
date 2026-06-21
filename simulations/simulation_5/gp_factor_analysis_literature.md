# GP factor analysis — literature & identifiability (research note)

*Compiled 2026-06-21. Context for the GP-GLLVM / ZQE line of work (`simulation_5/`,
`playground/gp-gllvm/`). The question that prompted this: does GP factor analysis already
exist, what are its identifiability results, and (in particular) does anyone estimate a
**cross-latent correlation** B?*

## TL;DR

- **GP factor analysis is a mature field.** The canonical method is **GPFA** (Yu et al. 2009).
  Many variants: Bayesian/variational, count-data (Poisson / negative-binomial / copula),
  multivariate (Duncker–Sahani).
- **Every standard method assumes the latent GPs are mutually independent (B = I).** The
  cross-latent correlation is *not a separate estimand* — it is **absorbed into the loadings**
  (a gauge). This is exactly the `W ↔ chol(B)` non-identification we derived independently.
- **Identifiability is obtained by:** (1) fixing the latent **marginal variance** `K_i(t,t)=1`
  (removes the X↔C scale gauge); (2) assuming **independent latents**; (3) **distinct per-latent
  timescales**, which break the residual rotation ambiguity. With a *shared* timescale the model
  is rotation-degenerate (pure gauge) — we reproduced this exactly (ℓ=[2,2] ⇒ ρ unidentified).
- **Consequence for us:** estimating a genuine cross-latent B is *non-standard*; the wall we hit
  (gauge + instability) is the known reason the field assumes independence. A genuine B needs
  distinct **cross-temporal** kernels (cross-spectral / convolved GPs) or hierarchical priors —
  a small, delicate literature. **Decision (2026-06-21): assume B = I and estimate per-latent
  timescales + loadings** (the standard, identified estimand), and make the contribution
  *scale* (Poisson, p up to ~10k, n in millions, likelihood-free via ZQE) rather than B.

## GPFA — the canonical model (Yu, Cunningham, Shenoy, Sahani 2009, J. Neurophysiol.)

Model: `y_t = C x_t + d + noise`, with `q` latent dimensions, each latent `x_{i,:}` an
independent GP over time (squared-exponential kernel, its own timescale `τ_i`). Gaussian
observations; EM for `C, d, R, τ_i`. Used pervasively in neuroscience for single-trial neural
trajectories.

Identifiability (direct quotes from the paper):
- Scale gauge: *"the scale of X is arbitrary … any scaling of X can be compensated by
  appropriately scaling C."* → resolved by **fixing `K_i(t,t)=1`** (signal variance 1;
  `σ_f,i² = 1 − σ_n,i²`, GP noise `σ_n,i² = 1e-3`).
- Latents: `x_{:,t} ~ N(0, I)` — **independent across dimensions a priori (B = I)**; correlation
  across *time* comes from the GP, correlation *across neurons* is stored in `C`.
- Each dimension gets a **distinct timescale** `τ_i`; rotation degeneracy resolved post-hoc by
  **orthonormalizing C** (and ordering by variance explained).

Implementations: **`elephant.gpfa`** (Python, maintained; expects `neo.SpikeTrain`, Gaussian via
sqrt-transform of counts); standalone repos `aecker/gpfa`, `wrongu/gpfa`.

## Variants

- **Count-data GPFA.** Conditionally-conjugate GPFA for spike counts via data augmentation
  (negative-binomial / binomial), 2024 — *also imposes independence over latent processes "for
  tractability"*; rotation handled by post-hoc orthonormalization. Poisson/NB likelihoods.
- **Bayesian GPFA with copula for count data** (Expert Syst. Appl., 2022) — copula to couple
  count margins.
- **Variational / scalable GPFA**, **Duncker & Sahani 2018** (temporal alignment + latent GP
  inference), multivariate GP factor models.
- **MOGP / coregionalization (LMC, SLFM, convolved GPs).** "Output correlation" is encoded in the
  **mixing matrix**, i.e. outputs are linear mixes of *independent* latent GPs — again, no
  separate latent-correlation parameter. Genuine cross-latent correlation needs a distinct
  **cross-spectral / convolved** cross-kernel (what our `build_Sigma` cross-block does).
- **Identifiable nonparametric factor analysis (NIFTY, 2023)** and **structured latent factor
  models with GP priors (2025)** — confirmatory/structural constraints (fixed zeros, anchors) to
  make factors identifiable.

## How this matches our own findings

- **`W ↔ chol(B)` gauge:** lower-triangular `W` removes the **O(q) rotation** gauge but NOT the
  `W↔chol(B)` gauge (because `A·C⁻¹` stays lower-triangular). Confirmed: with lower-tri W, ρ still
  unidentified from instantaneous structure.
- **Distinct kernels identify (in principle):** shared kernel ℓ=[2,2] ⇒ ρ is a pure gauge (stays
  at init, scatters); distinct kernels ⇒ ρ becomes identified (tight) — matches GPFA's
  "distinct timescales break rotation."
- **Residual bias is separate:** with distinct kernels ρ is *identified but biased high* (≈0.9,
  worsening with distinctness) — the encoder self-consistency bias, NOT identification. See
  `playground/gp-gllvm/gp_gllvm_bias.ipynb`.

## Implications / decisions

1. **Assume B = I, estimate per-latent timescales (+ loadings).** This is the identified,
   standard estimand. Our edge is the **likelihood-free ZQE at scale** (non-Gaussian, huge p, huge
   n via K-subset / marginal theorem) where EM-GPFA (O(G·T³), Gaussian) cannot go.
2. If a genuine cross-latent B is ever revisited: only identifiable via distinct cross-temporal
   kernels; expect delicate estimation and the encoder bias; would need a non-imputation moment.
3. **Baseline to run on our simulation:** standard GPFA (independent latents) — should recover the
   timescales and the *effective* loadings `A = W·chol(B)`, demonstrating the absorption of B.

## Sources

- Yu, Cunningham, Santhanam, Ryu, Shenoy, Sahani (2009), *Gaussian-Process Factor Analysis for
  Low-Dimensional Single-Trial Analysis of Neural Population Activity*, J. Neurophysiol.
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC2712272/
  - https://journals.physiology.org/doi/full/10.1152/jn.90941.2008
- Elephant GPFA (implementation + tutorial): https://elephant.readthedocs.io/en/latest/tutorials/gpfa.html
  ; reference: https://elephant.readthedocs.io/en/v0.7.0/reference/gpfa.html
- GPFA repos: https://github.com/aecker/gpfa ; https://github.com/wrongu/gpfa
- Conditionally-Conjugate GPFA for spike-count data (2024): https://arxiv.org/html/2405.11683
- Bayesian GPFA with copula for count data (2022): https://www.sciencedirect.com/science/article/abs/pii/S0957417422001336
- Duncker & Sahani (2018), temporal alignment + latent GP inference:
  http://www.gatsby.ucl.ac.uk/~maneesh/papers/duncker-sahani-2018-nips.pdf
- Gaussian orthogonal latent factor processes (2020): https://arxiv.org/pdf/2011.10863
- Identifiable & interpretable nonparametric factor analysis / NIFTY (2023): https://arxiv.org/abs/2311.08254
- Bayesian nonlinear structured latent factor models w/ GP prior (2025): https://arxiv.org/html/2501.02846
- Multi-output GP / coregionalization (LMC) overview (PyMC):
  https://www.pymc.io/projects/examples/en/latest/gaussian_processes/MOGP-Coregion-Hadamard.html
- Convolved MOGP for dependent count data (2017): https://arxiv.org/pdf/1710.01523
- Computationally efficient convolved MOGP, Álvarez & Lawrence (2011, JMLR): https://www.jmlr.org/papers/volume12/alvarez11a/alvarez11a.pdf
