# Future paper idea: GP-LVM via Z_q estimating equations

## One-line pitch

Apply the Z_q / ZQE framework to **Gaussian-process latent variable models**
(GP-LVMs), replacing intractable marginal-likelihood or ELBO optimisation with
a parameter-free surrogate-posterior estimating equation — inheriting all the
consistency, large-p concentration, and amortisation-gap-free properties of
the current theory, while adding structured (temporal, spatial, developmental)
latent space.

---

## Background and motivation

A GP-LVM (Lawrence 2005) posits:

    Z ~ GP(0, K_ψ)        prior over n latent vectors, indexed e.g. by time/space
    Y_i | z_i ~ Poisson(exp(W z_i + b))    (or any GLM family)

The latent Z carries **correlation structure** across observations — cells along
a developmental trajectory, spatial transcriptomics spots, longitudinal
measurements — which is precisely what standard i.i.d. GLLVM discards.

**The problem with existing approaches:**
- Marginal likelihood p(Y; θ, ψ) = ∫ p(Y|Z) p(Z; ψ) dZ is intractable for
  non-Gaussian likelihoods.
- GP-VAE (Fortuin et al. 2020, NeurIPS): wraps a VAE encoder around the GP
  prior. Still has the amortisation gap. The encoder now has to learn to map
  Y → Z while respecting GP covariance — even harder than i.i.d. case.
- Sparse GP approximations (inducing points, SVGP) reduce O(n³) to O(nm²)
  but still optimise an ELBO. All the known problems remain.

---

## The Z_q idea for GP-LVMs

### Surrogate posterior

For a GP prior Z ~ N(0, K_ψ) (K_ψ is the n×n kernel matrix, ψ = kernel
hyperparameters) and a GLM likelihood, the exact posterior p(Z|Y; θ, ψ) is
intractable, but a **Gaussian proxy** is:

    q(Z | Y; θ, ψ)  =  N(μ_q, Σ_q)

where (for a linear approximation to the Poisson link):

    Σ_q^{-1}  =  K_ψ^{-1}  +  W^T Λ W          (Λ = diagonal precision from GLM)
    μ_q       =  Σ_q  W^T  T(Y)                  (same T(y) as in the GLLVM paper)

This is the GP-LVM analogue of the Gaussian proxy encoder in the current paper.
**It has zero learnable parameters.** It is computed analytically given (θ, ψ).

Crucially:
- The large-p concentration result **transfers unchanged**: μ_q(y_i) involves a
  sum over p features via W^T T(y_i), which concentrates at rate 1/√p as before.
- The GP prior covariance K_ψ now couples the z_i's — the surrogate samples
  Z_q ~ q(Z|Y) correctly account for inter-observation correlation.

### Estimating equations

Define the joint surrogate sample Z_q ~ q(Z|Y; θ, ψ) and the Z_q estimating
equations exactly as in the GLLVM paper:

    Ψ_n(θ, ψ)  =  E_{q}[T(Y) ⊗ η(Z_q)]  −  E_{f_{θ,ψ}}[T(Y) ⊗ η(Z_q)]  =  0

where the second expectation is over (Y_q, Z_q) ~ p(Z; ψ) p(Y|Z; θ).

The parameters θ = (W, b) are the loading/bias as before.
The kernel hyperparameters ψ = (ℓ, σ²_f, σ²_n) enter through K_ψ in both the
surrogate and the generative draw.

**Theoretical questions to address:**
1. Does consistency / asymptotic normality of Theorem 1 extend to this joint
   (θ, ψ) system? The GP prior couples observations, so the i.i.d. assumption
   in the current proof breaks — need a mixing / ergodic argument.
2. What is the right notion of "large p" here? The loading W is still p×q, so
   the same 1/√p concentration applies to the per-observation part; the GP
   coupling adds an n×n layer on top.
3. Can sparse GP (inducing points) be plugged in as the surrogate without
   affecting consistency? Analogous to the current paper's result that any
   surrogate works as long as it's θ-consistent.

### Why this is better than GP-VAE

| | GP-VAE | GP-ZQE (proposed) |
|---|---|---|
| Encoder params | O(p × hidden) | **0** |
| Amortisation gap | ✓ present | **absent by construction** |
| Scales with p | ✗ degrades | **✓ concentrates at 1/√p** |
| GP structure used | in prior only | in both prior and surrogate |
| Training objective | ELBO (biased) | moment equations (unbiased) |
| Kernel hyperparams | ELBO-tuned (biased) | Z_q-tuned (consistent) |

---

## Key application: single-cell developmental trajectories

scRNA-seq with pseudotime:
- p ~ 20,000 genes (large-p regime — concentration is essentially exact)
- n ~ 5,000–50,000 cells
- Structure: cells lie on a low-dimensional developmental manifold, indexed by
  pseudotime t_i ∈ [0,1].

Model:
    K_ψ(i,j) = σ²_f exp(−|t_i − t_j|² / 2ℓ²)   (RBF kernel on pseudotime)
    Y_ij | z_i ~ Poisson(exp(W z_j + b_j))        (gene j, cell i)

The GP-ZQE estimator would:
1. Take pseudotime t (from e.g. diffusion pseudotime or Monocle) as given.
2. Fit W, b, ℓ, σ²_f by solving the Z_q moment equations.
3. Produce a **smooth** latent trajectory Z(t) with uncertainty quantification.
4. The large-p concentration means the proxy posterior is essentially exact
   even with n=5000 cells — no encoder training required.

Compare to: scVI (VAE, no GP structure), GPFA (Gaussian FA, no Poisson), 
GPLVM-DGP (deep GP, intractable), Gaussian process latent variable models for
single-cell (very recent, still ELBO-based).

---

## Connection to existing theory in the current paper

- §3 (Z_q estimating equations): applies verbatim with GP prior substituted
  for i.i.d. N(0,I). The centering structure is identical.
- §4.2 (Gaussian proxy): the proxy formula above is the GP generalisation of
  eq. (MAP proxy). The "Gaussian FA efficiency" result (Appendix C) may extend
  to GP-FA (linear W, GP prior, Gaussian likelihood) — worth checking.
- §5 (large-p regime): the 1/√p concentration argument is about W^T T(y_i),
  which is per-observation. The GP coupling is an additional n×n layer that
  doesn't affect the per-feature sum. So the large-p benefit is inherited.
- §6 (GMM overidentification): stacking multiple T_k still gives an
  overidentified GMM; optimal weighting now involves the GP-correlated
  residuals — potentially more interesting sandwich variance structure.

---

## Open questions / risks

1. **Pseudotime is not observed**: in practice t_i must be estimated. Does
   plugging in an estimated t̂_i affect consistency? This is a generated
   regressors problem.
2. **Non-stationary trajectories**: branching, multiple lineages — RBF kernel
   is too simple. Would need tree-structured or manifold-valued GP priors.
3. **Identifiability**: W is identified only up to rotation in the GLLVM; the
   GP structure may help by constraining the trajectory, but needs careful
   treatment.
4. **Computational cost**: forming K_ψ is O(n²), applying it is O(n³). Need
   sparse / inducing-point GP. The surrogate q(Z|Y) then becomes a sparse GP
   posterior — still closed-form, still zero encoder parameters.

---

## Draft title ideas

- "Z_q estimation for Gaussian-process latent variable models"
- "Parameter-free inference in GP-LVMs via surrogate-posterior estimating equations"
- "Amortisation-free GP-LVM for single-cell genomics"

---

## Status: idea only — not started

This note is a thinking aid. Nothing is implemented. The current paper should
be finished first. This is a natural follow-up that reuses essentially all of
the theoretical machinery already developed.
