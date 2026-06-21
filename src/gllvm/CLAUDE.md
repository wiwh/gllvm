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

## autofit.py  ← recommended ZQE entry point

### `ZQEAutoFitter(gllvm, *, encoder_factory=None, ...)`
Automatic ZQE fitter.  Family-agnostic: only uses `gllvm.sample`, `gllvm.zq_log`, and a
parameter-free `encoder.sample` (default `MapEncoderGaussianLog1p`).

Recipe (single chain — **no heads**; the old multi-head/consensus design was removed because it
risks OOM, scales poorly, and its cross-head spread is *algorithmic* noise, not a standard error):
1. **Warm-up (Adam)** on the decoder, LR annealed by `ReduceLROnPlateau` on per-epoch grad-norm;
   exits at the LR floor (`warmup_min_lr`).
2. **Refine (SGD + Ruppert–Polyak)**: one chain whose LR **decays within the chain**
   (`lr_k = refine_lr / (1+k)**refine_lr_power`, `power=0.5` classic; `0` → constant) with a
   Polyak tail average (uniform by default; EMA if `ema_decay` set).  Decreasing LR + tail
   averaging is the √n-optimal estimator and removes the O(lr) noise floor *at its source*
   (the variance-reduction step).
3. **Sequential-restart check**: each round warm-restarts the chain from the current Polyak
   estimate at a **decayed** LR (`refine_lr * refine_lr_decay**round`); `change_ =
   procrustes_error(prev, new)` (O(q)-aligned) measures how far it moved.  `change_ < tol` →
   converged.  Annealing matters: constant-LR Polyak has an **O(lr) residual noise floor**, so
   without decay the restart change can't fall below ~lr and never reaches `tol` (this is the
   fix for fits that "stop" instead of converging).  Watch `lr·‖avg ∇W‖/‖W‖` (the iterate's
   per-step drift) shrink alongside `change_`.

Also exports `procrustes_error(W_true, W_est)` (relative orthogonal Procrustes error, the
project's loadings metric; accepts numpy **or** torch, so it compares a `GLLVM.wz` against R's
loadings) and `orthogonal_align(ref, W)` (the underlying O(q) rotation).

Outputs: `.model` (Polyak-averaged `GLLVM`), `.change_` (Procrustes change at the last restart),
`.grad_norm_` (‖tail-avg ∇W‖/‖W‖ over the Polyak window — the empirical estimating equation; ≈0
at a root, a stationarity check distinct from the objective *value*), `.avg_grad_wz_` ((p,q)
tail-averaged loading gradient), `.converged_`, `.n_rounds_used_`, `.y_` (data, for post-hoc
deviance), `.history`.  **The estimate is the Polyak average** (`AveragedModel.module`, the tail
mean — *not* the last iterate); Procrustes and every diagnostic read it.  Simple knobs:
`warmup_lr`, `refine_lr`, `refine_lr_power` (within-chain Ruppert–Polyak decay exponent),
`refine_lr_decay` (base-LR multiplier *between* restarts; both annealings turn "stopped" into
"converged"), `steps_per_round`, `max_rounds`, `tol`, `warmup_optimizer`/`refine_optimizer`
(str key or optimiser class).  History adds `refine_lr` (per-step within-chain schedule, plotted
by `plot_lr`); `round_lr` is the tail-mean (effective) LR used for the `lr·grad` drift.

`.history` is rich enough to *replay* the fit (consumed by `diagnostics.py`): per warm-up epoch
`warmup_loss/gnorm/lr/wz/bias`; per restart `round_change/lr/grad_norm`, `refine_loss/gnorm`
((steps,) per-step traces — so the objective can be seen fluctuating about 0), and `round_wz/bias`
Polyak snapshots.

---

## diagnostics.py  ← plots that *show* the fit behaving

Functions take a fitted `ZQEAutoFitter` and read `.history`; nothing re-runs the fit.  Each
accepts an optional `ax`.  `plot_objective` is the headline check — the ZQE loss `-(m1-m2)` is
the negated empirical estimating equation, so it must **fluctuate about 0** at convergence (a
biased optimiser sits off-zero).  `plot_grad_balance` is the parameter-space twin: the
tail-averaged loading gradient (≈0 for most loadings at a root — instantaneous `plot_gradnorm`
stays noisy, but the *average* cancels to ≈0).  Also `plot_lr`, `plot_gradnorm`, `plot_convergence`
(per-restart Procrustes change, `grad_norm`, **and** `lr·grad_norm` = the iterate's per-step
drift, vs `tol`), `plot_params` (random-loading
trajectories, rotation-aligned to a common gauge; pass `g_true` for dashed true-value targets),
`plot_deviance` (Poisson deviance/obs, recomputed post-hoc from snapshots — assumes log-link
Poisson), and `plot_diagnostics` (2×4 dashboard).  Imports matplotlib, so it is **not**
re-exported from `gllvm/__init__.py`; use `from gllvm import diagnostics`.

---

## r_gllvm.py  ← R `gllvm` baseline wrapper

### `RGllvm(rscript=..., workdir=..., method="VA", family="poisson", maxit=2000, ...)`
Thin Python wrapper around R's `gllvm::gllvm()` (Niku et al.), run via `Rscript` in a
subprocess with CSV exchange.  `RGllvm(...).fit(Y, num_lv) -> RGllvmFit` with `.loadings` (p,q)
= `theta` scaled by `sigma.lv`, directly comparable to a Python `GLLVM.wz`.  `.available()`
checks the `Rscript` path before calling.  Defaults target WSL2→Windows R (`/mnt/c/...`,
auto-translated to `C:/...`); for a native install pass `rscript="Rscript"` and any writable
`workdir`.  Replaces the ad-hoc `run_r_gllvm()` helper that used to live in the notebooks.

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
- `._T_canonical(y)` → family canonical statistic (abstract; `y` for Poisson, `log(y)` for Gamma)
- `.T(y)` → returns the `T=` override if passed to `__init__`, else `_T_canonical(y)`
- `.zq_log(y)` → `T(y) * eta()` — used by `GLLVM.zq_log()`

**Configurable T**: pass any callable as `PoissonGLM(linpar, T=torch.log1p)`.  Persist it on a
model via `add_glm(PoissonGLM, idx=..., params={"T": torch.log1p})` (the override is stored in
`GLMFamily.params` and survives the transient re-instantiation in `forward`/`sample`/`zq_log`,
and `deepcopy`/`AveragedModel`).  Sampling and log_prob are **unchanged** by T — the ZQE only
changes the estimating function, not the generative model.  The old subclass-per-T zoo
(`PoissonLog1pGLM`, `PoissonSqrtGLM`, `PoissonMixedTGLM`, `PoissonLog1pSqrtGLM`,
`PoissonMultiTGLM`) was removed; re-express as `T=<callable>`.

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

---

## kernels.py  ← GP-GLLVM latent covariance

### `Kernel` / `RBFKernel(latent_dim, lengthscale=1.0, jitter=1e-4)`
Per-factor covariance for the GP-GLLVM latent prior.  Because the factors are
**independent (`B = I`)**, the latent covariance is **block-diagonal**: a kernel maps
coordinates `(*batch, K, d)` to the `q` per-factor blocks `(*batch, q, K, K)` via
`.blocks(coords)`, with `.cholesky(coords)` derived.  `RBFKernel` is squared-exponential
with a **per-factor** length-scale `ell_k` (stored as `log_lengthscale`, a `Parameter`);
distinct `ell_k` are what break the rotation gauge and identify the loadings.

## gpgllvm.py  ← GP-GLLVM model + encoder + fitter

### `GPGLLVM(latent_dim, output_dim, *, input_dim=1, kernel=None, lengthscale=1.0, jitter=1e-4, …)`
**Subclasses `GLLVM`** — same decoder (`add_glm`, `forward`, `log_prob`, `zq_log`, families),
but the latent prior is a per-factor GP over coordinates instead of i.i.d. `N(0,I)`.
Defaults to `float64` (Cholesky stability).  Key methods:
- `.sample_z(coords)` → GP draw `(*batch, K, q)`, exact per-factor (`z[...,k]=L_k ε_k`).
  (Overrides `GLLVM.sample_z(num_samples)` — the GP prior needs coordinates.)
- `.sample(coords=…, z=…, offset=…)`, `.forward(z, x, offset=…)` — **offset** is a known
  additive term in η (e.g. log library size), broadcast over responses.
- `.whiten(z, coords)` = `L_Σ⁻¹ z`; `.cov(coords)` = block-diagonal `Σ` `(*batch, qK, qK)`;
  `.lengthscale` property.
- Works for any leading batch shape (patches), `input_dim` 1 (time) or 2 (space) or more.

### `GPMapEncoder(gpgllvm, sigma2=1.0)`
Parameter-free **joint block-MAP** imputer under the Gaussian-`log1p` proxy: solves
`(Σ⁻¹ + (WᵀW/σ²)⊗I_K) vec(z) = vec((log1p(y)-b-offset)W/σ²)`.  `Σ⁻¹` built per factor
(block-diagonal); `.sample(y, coords, offset)` → `(ẑ, ẑ, -inf)` (δ-mass, detached by the
fitter via the score-function identity).  **Must encode the whole K-block jointly** — the
joint prior is what makes the imputed `ẑ` carry the kernel structure the ℓ-fit needs.

### `GPZQEFitter(model, *, encoder=None, K=64, steps=2000, lr=0.03, batch=128, warmup=200, fit_lengthscale=True, cov_batch=32, …)`
Likelihood-free ZQE fitter.  `.fit(y, coords, groups=None, offset=None)` →
`self` (`self.model` fitted in place; `.history`, `.lengthscales_`, `.fit_time_`).
Each step samples a random **K-subset of each group** (GP marginal theorem → every op is
`K×K`, cost independent of group size) and minimises
`-(m1-m2)` (centered loadings, per-obs) **+** a fantasy-centered second-moment term
`‖Σ(ℓ) − E[ẑẑᵀ]‖²` per patch (the cross-observation moment that identifies ℓ).
`warmup` steps fit loadings only; `fit_lengthscale=False` freezes ℓ (e.g. a fixed-ℓ scan).
`W` is kept **full** (distinct ℓ_k identify it — no lower-tri; see `paper/CLAUDE.md`).
Verified: synthetic recovery ℓ̂≈[0.96,3.11] for true [1,3], procW≈0.05 over seeds
(`playground/gp-gllvm/_verify_gpgllvm_api.py`).

**GP gotchas.**  (a) Use the default `float64`; `float32` Cholesky is fragile with small
`jitter`.  (b) `K` is capped to the smallest group size.  (c) `cov_batch` bounds the
second-moment term's memory (`O(cov_batch·(qK)²)`); lower it for large `q`.  (d) The encoder
solve is dense in `qK` — fine for the `q` we use; the block-diag/CG large-`q` variant is
deliberately deferred (Paper 2, prototype `playground/gp-gllvm/_blockscale_verify.py`).
