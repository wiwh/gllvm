# CLAUDE.md — gllvm project context

## What this repo is

A PyTorch implementation of **Generalized Latent Variable Models (GLLVMs)** with a novel
estimation method called **ZQE (Z-estimating equations)**.  The primary scientific goal is to
show that ZQE can recover loading matrices more accurately than VAE-based training, in particular
under sparse loadings (single-active-factor structure).

The repo is an **editable package install**: `pip install -e .` in the workspace root.
All imports use the `gllvm.*` namespace from `src/gllvm/`.

---

## Core estimation methods

### VAE (baseline)
Standard variational autoencoder:  amortised encoder `q_φ(z|y)` + decoder `p_θ(y|z)`.
Trained jointly by maximising the ELBO.  **Fitter**: `VAEFitter` in `src/gllvm/fitter.py`.

### ZQE (proposed method)
Z-**estimating equation** approach.  For a Poisson log-link GLLVM with η = Wz + b:

    E[T(y) · η(z_q(y); θ)] − E_θ[T(Y_q) · η(z_q(Y_q); θ)] = 0

The loss implemented in the notebook is `-(m1 - m2)` where:
- `m1 = E[T(y) · η(z_enc(y))]`   — real-data term
- `m2 = E_θ[T(Y_q) · η(z_enc(Y_q))]`  — model-data centring term (removes bias)

**Key insight (score-function identity)**: The gradient of this equation w.r.t. θ does NOT
require gradients through the encoder.  Detaching the encoder inside `torch.no_grad()` is
**exact**, not an approximation.  This means parameter-free (analytical) encoders work just as
well as amortised ones — and are strictly better because they update automatically with θ.

**T(y) choice**: T is a per-instance callable configured at construction —
`PoissonGLM(linpar, T=torch.log1p)` (threaded through `GLMFamily.params`, so
`add_glm(PoissonGLM, idx=..., params={"T": torch.log1p})`).  It defaults to the family
canonical statistic (`_T_canonical`, = `y` for Poisson) when omitted.  Use `T=log1p` for
ZQE: the canonical `T=y` causes NaN/explosion when the decoder is randomly initialised
(large Poisson rates in synthetic Y_q); `log1p` bounds the centring term.  Any measurable
T is valid (score-function identity), so T is decoupled from the generative model.

**Fitter**: `ZQEAutoFitter` in `src/gllvm/autofit.py` — the recommended entry point.  Adam
warm-up → single-chain SGD + **Ruppert–Polyak** averaging (LR decays *within* each chain,
`lr_k = refine_lr/(1+k)**refine_lr_power`; the variance-reduction step), with a
**sequential-restart** convergence check: warm-restart at a further-decayed base LR
(`refine_lr_decay` per restart), Procrustes-align, and stop when the estimate stops moving
(`change_ < tol`).  LR annealing is essential — constant-LR Polyak has an O(lr) noise floor, so
without it `change_` plateaus above `tol` and the fit "stops" instead of converging.  A
complementary check is the tail-averaged gradient `grad_norm_` = ‖avg ∇W‖/‖W‖ ≈ 0, and its
per-step drift `lr·grad_norm_` (both in `plot_convergence`).  (The earlier multi-head/consensus
variant was dropped — heads OOM and scale poorly, and their spread is *algorithmic* noise, not a
standard error.)  The older hand-rolled fitters in `fitter.py` (`ZQEPoissonFitter`, `ZQEFitter`,
…) are superseded.

---

## Encoder taxonomy

All encoders expose the same interface: `.sample(y) → (z, mu, logvar)`.

| Class | Type | Parameters | Notes |
|-------|------|-----------|-------|
| `Encoder` | Amortised VAE encoder | Learnable NN | For VAE training |
| `MapEncoderGaussianLog1p` | Parameter-free MAP | None | `z_MAP = (W^TW + σ²I)^{-1} W^T (log1p(y)-b)` |
| `GaussianPosteriorEncoderLog1p` | Parameter-free posterior sampler | None | Draws `z ~ N(μ_MAP, σ²(W^TW+σ²I)^{-1})`; Cholesky reparameterisation |

The two parameter-free encoders always use the **current** θ = (W, b).  This is why they are
consistent from random initialisation — the encoder "tracks" the decoder automatically.

### Why frozen amortised encoders fail

A frozen `Encoder` (trained at θ_VAE) is only a consistent z-estimator near θ_VAE.  When the
decoder θ drifts, the encoder still maps y → z according to θ_VAE.  The centring term
`E_θ[T(Y_q)·η(z_enc(Y_q);θ)]` uses a mismatched encoder, so the estimating equation is
**biased** at any θ ≠ θ_VAE.  Gradient descent exploits this bias: the ZQE loss becomes
unbounded below, wz entries blow up, Procrustes error → 1.

**Corrective experiment**: start the decoder at θ_VAE when using a frozen encoder, so it is
initially in the region of encoder validity.

---

## GLM families

Defined in `src/gllvm/glms.py`.  All inherit from `torch.distributions.*` AND `GLMMixin`.

- `PoissonGLM`          — log link; canonical `T(y)=y`, pass `T=torch.log1p` for ZQE
- `GaussianGLM`, `GammaGLM`, `NegativeBinomialGLM`, `BinomialGLM`

Each family defines `_T_canonical(y)` (its canonical statistic); the public `T(y)` returns the
`T=` override if one was passed to `__init__`, else `_T_canonical`.  This replaced the old
subclass-per-transform zoo (`PoissonLog1pGLM`, `PoissonSqrtGLM`, `PoissonMixedTGLM`, …), which
has been removed — express any of them as `PoissonGLM(..., T=<callable>)`.

The `zq_log(y)` method returns `T(y) * eta()` — the informative term in the estimating equation.

---

## GLLVM module (`src/gllvm/gllvm_module.py`)

`GLLVM(latent_dim, output_dim)` stores:
- `wz`: loadings matrix `(p, q)` — the main parameter of interest
- `bias`: intercept vector `(p,)`
- `log_scale`: dispersion `(p,)` — usually irrelevant for Poisson
- `families`: list of `GLMFamily` objects assigning GLM classes to response subsets
- `wz_mask`: optional structural-zero mask `(p, q)` for sparse loadings

Key methods: `.forward(z)`, `.sample(z=...)`, `.log_prob(y, z=...)`, `.zq_log(y, z=...)`

---

## Simulations

### `make_sparse(n_latent, poisson, active_latent, wz_scale)`
Creates a GLLVM where each latent dimension is "active" for only `active_latent` randomly
chosen features (structural zeros elsewhere).  Used in `simulation_sparse/`.

### `make_mixed(n_latent, poisson, wz_scale)`
Creates a GLLVM with all Poisson features and no structural zeros.  Used as the VAE decoder
template (which is then re-randomised by `fresh_decoder()` for ZQE arms).

---

## Key experimental results (simulation_sparse, seed=42, NL=3, ACT=1, NR=50, NS=250)

| Method | Procrustes error |
|--------|-----------------|
| ZQE MAP | ~0.48 |
| ZQE Gauss-post | ~0.50 |
| VAE | ~0.74 |
| ZQE Frozen-VAE (random init) | ~0.91 (diverges) |
| ZQE Frozen-VAE (VAE init) | ~0.775 — still diverges, wz → ±50 |

ZQE with parameter-free encoders beats VAE by ~35% on Procrustes error.

---

## Evaluation metric: orthogonal Procrustes distance

All methods are compared using the **relative orthogonal Procrustes error**:

$$d(W, \hat{W}) = \frac{\min_{R \in O(q)} \|W - \hat{W} R\|_F}{\|W\|_F}$$

where $R^* = VU^T$ from the SVD $W^T \hat{W} = U S V^T$.

**Important**: we do NOT use `scipy.spatial.procrustes`, which applies centering (subtracts
column means) and scales both matrices to unit Frobenius norm before comparing. Those
transformations are wrong for loading matrices:
- centering is meaningless — columns of $W$ are not naturally zero-mean
- equal-norm scaling means a shrunken/biased estimate scores the same as a perfectly-scaled one

The correct metric is pure orthogonal rotation only. Division by $\|W\|_F$ makes the score
scale-free: a method that shrinks all loadings toward zero will score worse (correctly).

Implemented as `procrustes_error(W_true, W_est)` in `gllvm.autofit` (accepts numpy or torch,
so it scores ZQE, VAE and the R baseline uniformly).  Older notebooks inline an equivalent
`procr(g_true, g_est)`.

---

## File map

```
src/gllvm/
  encoder.py          — all encoder classes
  autofit.py          — ZQEAutoFitter (recommended ZQE fitter) + orthogonal_align + procrustes_error
  diagnostics.py      — plot_objective/lr/gradnorm/convergence/grad_balance/params/deviance (reads fitter.history)
  r_gllvm.py          — RGllvm: subprocess wrapper around R's gllvm() (the baseline)
  fitter.py           — VAEFitter (+ superseded ZQE fitters: ZQEPoissonFitter, ZQEFitter, …)
  gllvm_module.py     — GLLVM model, GLMFamily
  glms.py             — GLM distribution classes (T configurable via PoissonGLM(..., T=…))
  simulations/        — make_sparse, make_mixed, simulate
  plots.py            — compare_wz, compare_bias, compare_z

simulations/
  poisson.ipynb                — clean 3-way Poisson comparison (Poisson-MAP / Gaussian-MAP / R gllvm)
simulations/simulation_sparse/
  exp_A_sparse_loadings.ipynb  — main experiment notebook
```
