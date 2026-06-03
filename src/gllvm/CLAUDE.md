# CLAUDE.md — src/gllvm source package

## Package overview

The `gllvm` package implements Generalised Latent Variable Models with two estimation strategies:
VAE (ELBO maximisation) and ZQE (Z-estimating equations).

---

## encoder.py

### `Encoder(input_dim, latent_dim, hidden=32)`
Standard amortised VAE encoder.  Two-layer ReLU MLP → (mu, logvar).
`.sample(y)` → `(z, mu, logvar)` via reparameterisation.
`.loss(y, gllvm)` → `(-elbo, elbo_scalar)` using GLLVM log_prob.

### `MapEncoderGaussianLog1p(gllvm, sigma2=1.0)`
**No learnable parameters.**  Analytical MAP under Gaussian proxy:
```
log1p(y) | z ~ N(Wz+b, σ²I),  z ~ N(0,I)
z_MAP = (W^TW + σ²I)^{-1} W^T (log1p(y) - b)
```
Always uses the *current* gllvm.wz and gllvm.bias (live reference).
`.sample(y)` returns `(z_map, z_map, -inf)` — deterministic.

### `GaussianPosteriorEncoderLog1p(gllvm, sigma2=1.0)`
**No learnable parameters.**  Samples from the exact Gaussian posterior:
```
q(z|y) = N(μ, Σ),  Σ = σ²(W^TW + σ²I)^{-1},  μ = Σ/σ² · W^T(log1p(y)-b)
```
Covariance only depends on W (not y) → compute Cholesky once per batch.
`L_Sigma` is upper-triangular; `z = mu + eps @ L_Sigma`.

**Why this beats MAP in practice**: draws from full posterior inject useful variance into the
ZQE gradient, preventing early collapse onto degenerate solutions.

---

## fitter.py

### `VAEFitter(gllvm, encoder, lr, device)`
Joint ELBO optimiser.  Single Adam over `encoder.parameters() + gllvm.parameters()`.
`.fit(y, epochs, batch_size, patience, verbose)` — early stopping on ELBO.

Typical training recipe (see experiment notebook):
1. Warm-up: `lr=1e-2`, 200 epochs, no patience
2. Fine-tune 1: `lr=3e-3`, patience=50, restore best wz/bias/encoder
3. Fine-tune 2: `lr=1e-3`, patience=80, restore best

### `ZQEFitter(gllvm, encoder, ...)`
Older class with separate enc/dec optimisers.  **Superseded** by the inline `run_zqe()` helper
in the experiment notebook, which is simpler and has per-arm LR tuning.

### `ZQEGAFitter`
Blockwise version for large p.  Uses `EncoderGaussianApprox` with pre-computed (X^TX)^{-1}.
Experimental, not used in current simulations.

---

## gllvm_module.py

### `GLLVM(latent_dim, output_dim, feature_dim=0, bias=True)`
Core model.  Parameters: `wz (p,q)`, `bias (p,)`, `log_scale (p,)`, optional `wx (p,k)`.

**Important**: `families` is a plain Python list (not `nn.ModuleList`).  Add GLM families with
`.add_glm(GLMClass, idx, params, name)` or set `g.families` directly.

`wz_mask`: `(p, q)` buffer for structural zeros.  Set via `.set_wz_mask(mask)`.

Key methods:
- `.forward(z)` → linpar `(n, p)`
- `.sample(z=...)` → `y` samples per GLM family
- `.log_prob(y, z=...)` → `(n, p)` log-likelihoods
- `.zq_log(y, z=...)` → `T(y) * eta` per response  ← ZQE core

### `GLMFamily(GLM, idx, params, name)`
Associates a GLM class with a set of response indices.  `idx` can be list/range/ndarray/tensor.

---

## glms.py

All classes inherit from both `torch.distributions.*` and `GLMMixin`.

Critical interface (GLMMixin):
- `.eta()` → natural parameter (= linpar for log-link)
- `.T(y)` → sufficient statistic
- `.zq_log(y)` → `T(y) * eta()` — used by `GLLVM.zq_log()`

**`PoissonLog1pGLM`**: only overrides `T(y) = log1p(y.float())`.  Sampling and log_prob are
**unchanged** (uses canonical Poisson).  This is intentional: the ZQE only changes the
estimating function T, not the generative model.

---

## simulations/

### `make_sparse(n_latent, poisson, active_latent, wz_scale)`
Returns a `GLLVM` with sparse loadings: each latent dim is active for
`active_latent` random features, zero elsewhere.

### `make_mixed(n_latent, poisson, wz_scale)`
Returns a dense-loadings `GLLVM` with all Poisson responses.

### `simulate(gllvm, n_samples, device)`
Returns `(y, z)` from the generative model.

---

## Common gotchas

1. **`g.families` is mutated in-place by `swap_log1p()`** — always call on a `deepcopy`.
2. **`enc_vae_m` takes raw counts** (not log1p) as input — it was trained that way.
   The analytical encoders apply `log1p` internally.
3. **`wz_mask` is a buffer**, not a parameter — it is included in `state_dict` and moves
   with `.to(device)`, but is not differentiated.
4. **GLMFamily idx must be on the same device as the model** — handled by `GLLVM.to()` override.
5. **NaN from random decoder**: when `wz` is randomly initialised, `exp(Wz+b)` can overflow
   for large |z|.  `PoissonGLM` clamps linpar at 10 (`exp(10)≈22k`).  Still, `y_q` can be
   large.  `T=log1p` avoids explosion in the centring term; canonical `T=y` does not.
