# Simulation 6 — encoder transfer: frozen encoder under decoder shift

The purest illustration of the main theorem. The Jacobian-cancellation identity
says the surrogate's derivative drops out of the first-order Jacobian, so decoder
consistency does **not** require the encoder to track θ. This experiment cashes
that out as a *transfer* statement: **calibrate an encoder on a source population,
freeze it, reuse it on a shifted target — ZQE still recovers the shifted decoder**,
while the naive "treat the imputed `ẑ` as observed" estimator is biased.

This refines the muddy Experiment C of the draft: we replace the VAE-coupled
encoder (whose joint training with θ_src confounds "encoder mismatch" with
"decoder change") by the **frozen analytic Gaussian encoder**, an unambiguous fixed
mechanism `q(z|y)`. That removes the confound and isolates the claim.

## Set-up (what "frozen" means)

In simulations 1–3 the Gaussian encoder is **live**: it reads the *current* θ each
step (`MapEncoderGaussianLog1p(g)` holds a reference to the decoder), so it
self-corrects and consistency is almost automatic. Here we deliberately **freeze**
it at the source parameter:

```
q_src(z | y) = (W_srcᵀ W_src + σ²I)⁻¹ W_srcᵀ (log1p(y) − b_src)     # fixed; never tracks θ
```

and keep using it while the *true* decoder is shifted to θ_tgt. The encoder is now
genuinely wrong for the target, which is exactly the non-trivial test. Code-wise
this is a one-line change: bind the encoder to a frozen copy `g_src`
(`encoder_factory = lambda g: MapEncoderGaussianLog1p(g_frozen)`), so the fitter
updates the decoder while the encoder stays at θ_src.

## Design

Dense Poisson GLLVM, `q=2, p=50, n=200` (as in simulation 2). Fix a source decoder
θ_src and build the frozen encoder from it. Generate the **target** data from a
shifted decoder θ_tgt obtained by perturbing the loadings — primarily one column
`W_{·,k} → W_{·,k} + Δ` — and sweep the **relative shift magnitude**

```
δ = ‖Δ‖ / ‖W_{·,k}‖  ∈  {0, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0}
```

(intercepts and prior fixed, so "shift" has one unambiguous meaning). At each δ,
fit the target decoder three ways and score recovery of θ_tgt.

## Methods (cheap+consistent vs cheap+biased vs expensive+consistent)

| method | encoder | cost | expected |
|--------|---------|------|----------|
| `zqe_frozen` | frozen at θ_src | cheap | **consistent** — recovers θ_tgt (the transfer claim) |
| `plugin_frozen` | frozen at θ_src, `ẑ` treated as **observed** → GLM | cheap | **biased** — pulled toward θ_src (this is the "as if we observed Z" estimator) |
| `gllvm_refit` | full VA refit on target | expensive | consistent reference (what you'd pay for without transfer) |

The headline: `zqe_frozen` matches the expensive `gllvm_refit` at the cost of the
biased `plugin_frozen`. (A live-encoder `zqe_track` run may be included as an upper
reference.) `plugin_frozen` also doubles as a **warm start**: `ẑ → GLM` lands near
θ, then a few ZQE centering steps remove the bias — fast *and* correct.

## What to show: flat until breakdown

- **Recovery vs shift.** Plot recovery of θ_tgt (overall Procrustes, and the
  perturbed column directly: estimated vs true new value) against δ. Expect
  `zqe_frozen` **flat near `gllvm_refit`** while `plugin_frozen` bias grows with δ.
- **The breakdown is identification, not bias.** `zqe_frozen` stays consistent
  while the frozen encoder's features remain informative about z under θ_tgt, i.e.
  while the Jacobian `A(θ_tgt)` stays nonsingular. At extreme δ (W_src no longer
  spans the target structure) it degrades through **variance/identification**, not
  bias — the same "flat until breakdown" shape as simulation 2, now in shift space.

## Why it matters

This is **transferable amortized inference**: amortize the expensive inference once
(the encoder), then fit decoders for many related datasets/populations cheaply,
each consistent despite the encoder being trained elsewhere — something `gllvm`
cannot do (it re-runs full VA per dataset). It is on-thesis (the encoder-agnostic
consistency theorem in action) and is the cleanest single demonstration of the
cancellation result.

## Reproducibility / layout

Same conventions as simulations 1–3: `seed == rep` drives θ_src, the shift, and
the target data; one CSV per (δ, rep); resumable; failures flagged. Driver:
`sweep.py` (to be written). Experiment: `experiment.ipynb`. Analysis:
`analysis.ipynb`. Columns: `results/DATA_DICTIONARY.md`.
