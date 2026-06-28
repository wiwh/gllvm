# Design note — matrix-free sandwich covariance for ZQE (scalable inference)

**Goal.** Give the ZQE loading estimates **calibrated standard errors at the scale ZQE is built for**
(p in the thousands–tens of thousands), where the CRN parametric bootstrap (`varboot.param_bootstrap_resolve`,
~7 s × D × H re-solves) is infeasible. This completes the "enabler" story: point estimation *and*
uncertainty quantification that both scale with p.

The bootstrap stays as the **validated ground truth** at moderate p (sim_8: SE/SD ≈ 0.99, loading
coverage ≈ 0.94). The deterministic sandwich must reproduce it at p=50, then run where the bootstrap
cannot.

---

## 1. The estimator and its sandwich

ZQE solves the empirical estimating equation `Ψ_n(θ) = (1/n) Σ_i ψ_i(θ) = 0`, where θ = (W, b) are the
*free* loadings + intercepts. It is a **Z-estimator** (a designed estimating equation, not the gradient
of a likelihood). Standard M/Z theory:

```
√n (θ̂ − θ0)  →  N(0,  A⁻¹ S A⁻ᵀ)
A = −E[∂ψ/∂θ]      (bread = Jacobian of the estimating function = loss Hessian)
S = Var(ψ(O; θ0))  (meat = covariance of the per-sample estimating function)
Cov(θ̂) ≈ (1/n) A⁻¹ S A⁻ᵀ
```

This is the Godambe / sandwich information. We never form the d×d matrices (d = #free params ≈ p·q);
everything is **matrix-free per functional**.

---

## 2. Two structural facts that make it cheap (ZQE-specific)

The ZQE loss is `L(θ) = m2(θ) − m1(θ)`, with
- `m1(θ) = (1/n) Σ_i T(y_i) · η(z_i; θ)`  — real-data term, `z_i = z_enc(y_i)` **detached**
- `m2(θ) = E_θ[ T(Y_q) · η(z_enc(Y_q); θ) ]` — model-fantasy centering, `Y_q ~ f_θ` (CRN-fixed seed)

and `η = W z + b`, `T = log1p`. Two consequences:

**(a) The data term is linear in θ → all curvature lives in the centering.**
For fixed detached `z_i`, `T(y_i)·η(z_i;θ)` is linear in (W,b), so `∇²m1 = 0`. Therefore

```
A = ∇²L = ∇²m2          (the bread is the centering Hessian alone)
```

This is the analytic restatement of "identification lives in the centering's θ-dependence."

**(b) The meat is a one-pass sample covariance of the data gradient.**
The per-sample estimating function is `ψ_i = c(θ) − g_i(θ)` where `c(θ)=∇m2` is **constant across i** and
`g_i(θ) = ∇_θ[T(y_i)·η(z_i;θ)]` has the closed form

```
∂/∂w_j  =  T(y_ij) · z_i        (a q-vector, per response j)
∂/∂b_j  =  T(y_ij)
```

Since `c(θ)` is constant in i, `S = Var_i(ψ_i) = Var_i(g_i)`. At θ̂, `Ψ_n=0 ⇒ mean_i g_i = c`, so

```
S = (1/n) Σ_i (g_i − ḡ)(g_i − ḡ)ᵀ ,   ḡ = (1/n) Σ_i g_i
```

`g_i` is `T(y_i)`-weighted `[z_i, 1]` per response — cheap and structured. We never store S; we only need
`wᵀ S w` for given vectors w (one data pass accumulating scalars `wᵀg_i`).

---

## 3. Per-functional variance (the matrix-free core)

For a scalar functional `g(θ)` (a free loading `W_jk`, or a Gram entry `(WWᵀ)_jk`):

```
Var(g(θ̂)) = ∇gᵀ Cov(θ̂) ∇g = (1/n) (A⁻ᵀ∇g)ᵀ S (A⁻ᵀ∇g)
```

Algorithm (per functional):
1. Form `∇g` (sparse: a loading functional touches ≤ 2 rows of W).
2. Solve `w = A⁻¹ ∇g` by **CG** (A is symmetric ⇒ A⁻ᵀ=A⁻¹), using only Hessian-vector products `A·v`.
3. One pass over data: `Var(g) = (1/n²) Σ_i (wᵀ(g_i − ḡ))²`.

Cost per functional ≈ (a handful of HVPs in CG) + (one O(n·d) data pass). **No d×d, no refitting, no D×H.**
Many functionals share the single data pass: precompute `w_k = A⁻¹∇g_k` for each, then accumulate all
`{w_kᵀ g_i}` in one sweep.

---

## 4. Hessian-vector product `A·v` (three options, all matrix-free)

`A = ∇²L` with the **fantasy held at a fixed (CRN) seed**, so `L(θ)` is a deterministic smooth function.

1. **Exact double-backward** (default): `A·v = ∇_θ( ⟨∇_θ L, v⟩ )` via torch double autograd. Encoder stays
   detached (`no_grad` / `.detach()` inside the loss, exactly as in the fitter), so we do **not**
   differentiate through `z_enc` — the score-function identity makes this exact.
2. **Finite-difference HVP** (gradient-only, if double-backward through the centering is awkward):
   `A·v ≈ [∇L(θ+εv) − ∇L(θ−εv)] / (2ε)`. Two gradient evals per HVP, no second-order autograd.
3. **Spall SP** (only if we ever want the explicit matrix or its diagonal): random Rademacher probes,
   `Â = avg symmetrized rank-1`. Not needed for CG; keep as fallback for a quick diagonal-only SE.

Validation should confirm (1) and (2) agree before trusting either at scale.

---

## 5. Gauge & conditioning

- **Work in the free lower-tri parameter space** (the `wz_mask` parameters). The lower-tri constraint
  removes the rotation null-space, so `A` restricted to free params is non-singular — this is exactly the
  parameterization R's `gllvm` uses for its observed-information SEs. **Our sandwich in this space is a
  scalable, matrix-free version of gllvm's Hessian SEs.** (Good framing for the paper.)
- The `c/n` **ridge is already in L**, contributing `+ (l2)·I` to `A` — it floors the near-null directions
  so `A` is PD and CG converges. Use it as the natural regulariser (and a Jacobi/`l2` preconditioner).
- For rotation-**invariant** functionals (`WWᵀ`), `∇g` is orthogonal to the gauge directions, so they are
  also well-defined; report these as the gauge-free cross-check.

---

## 6. Complexity / scaling

- meat: one O(n·d) pass per functional batch (shared).
- bread solve: `n_cg` HVPs per functional; each HVP is one (or two) loss-gradient evals ⇒ O(cost of one
  fitter gradient). `n_cg` small because the ridge conditions `A`.
- Total for K functionals ≈ `K · n_cg` gradient evals + one data pass. **Linear in p**, no D×H blow-up.
  Contrast: bootstrap = `D · H` full re-solves (each ~hundreds of gradient steps).

---

## 7. Validation plan

1. **Agreement at p=50** (tonight's setting, q=2, n=500, dense): compute sandwich SEs for the same
   functionals as `validation.npz`; check `sandwich_SE / bootstrap_SE ≈ 1` and that sandwich Wald
   coverage matches (≈0.94 loadings). This is the correctness gate (bootstrap = ground truth).
2. **Cross-check HVP exact vs finite-difference** (§4.1 vs §4.2).
3. **Scale demonstration**: p = 1k, 10k, 50k (reuse sim_7 truths). Show (a) wall-time ~linear in p,
   (b) SEs stable, (c) the bootstrap is infeasible there → the enabler claim is complete.

---

## 8. API sketch

New module `simulations/simulation_8/sandwich.py` (promote to `src/gllvm/inference.py` once stable):

```python
def loss_grad(ft, theta_vec) -> Tensor          # ∇L at θ, CRN fantasy fixed (reuse fitter internals)
def hvp(ft, theta_vec, v, *, mode="autograd") -> Tensor   # A·v  (autograd | fd)
def meat_apply(ft, w) -> float                   # (1/n²) Σ_i (wᵀ(g_i − ḡ))²   (one data pass)
def sandwich_se(ft, grad_g, *, l2, cg_tol=1e-6, cg_maxiter=200) -> float
    # w = CG(lambda v: hvp(ft, θ̂, v) , grad_g);  return sqrt(meat_apply(ft, w))
def loading_grad(p, q, j, k, mask) -> Tensor     # ∇ of W_jk in free-param coords
def gram_grad(W, j, k) -> Tensor                 # ∇ of (WWᵀ)_jk
```

Then a `coverage_experiment_sandwich(...)` mirroring the bootstrap one, writing SEs scored the same way
in `make_figure.py` (drop-in: the figure already reads SE arrays).

---

## 9. Risks / open issues

- **Double-backward through the centering**: the fantasy expectation `m2` may use a reparam/score-function
  estimator; second-order autograd must be valid. Mitigation: the FD-HVP (§4.2) is the safe fallback and
  needs only first-order grads.
- **CG conditioning** if `l2` is very small: monitor CG iters; bump preconditioner or `l2` floor if needed
  (it's the same near-unidentified directions the ridge already targets).
- **MC noise in the centering**: `A·v` and `g_i` depend on the fantasy draw. Use the **same CRN seed**
  throughout a sandwich computation (consistent with the bootstrap lesson). For S, the fantasy only enters
  through the constant `c`, which cancels — so S is fantasy-free; only `A` carries fantasy MC noise, reduced
  by a larger fantasy sample if needed.
- **Bias**: the `c/n` ridge gives O(1/n) shrinkage (already visible as the slight undercoverage); the
  sandwich inherits it. Acceptable and documented; optional Firth-style correction is future work.

---

## 10. Path-based / batch-means inference — tonight's findings (prototype `polyak_bm.py`)

The "free" alternative to the sandwich: read `Cov(θ̂)` off a Polyak-averaged SGD path via
batch-means (Chen–Lee–Tong–Zhang 2020; Zhu–Chen–Wu 2021). Mechanism is sound (the averaged-iterate
long-run variance *is* `A⁻¹SA⁻¹`; one fantasy draw per step supplies `S` by Fisher consistency,
since at the root the data and model sides of the equation are identically distributed). **But the
constant-LR prototype does NOT reproduce the bootstrap SEs**, and we found exactly why:

- **The chain doesn't stay stationary around `θ̂`.** It random-walks/drifts. Stability is
  **dataset-dependent**: d1 was clean (Polyak offset 2.4%, drift 4e-5) but d0/d3 drifted badly
  (offset 17–26%). The split tracks how flat the loss is for that data realization — a
  **weakly-identified direction** the sampler wanders along.
- **`θ̂` is only approximately a root** of the standalone chain (mean gradient at `θ̂` ≈ 9% of the
  per-draw noise, but `t≈5.5` → significantly nonzero) — a small persistent drift on top of the
  random walk.
- **The `c/n` ridge does NOT fix it.** A ridge sweep `c ∈ {0.001 … 100}` (10⁵×) changed offset,
  drift, and SE *not at all*: the loss normalizes the ridge by `/p/q`, so even `c=100` gives a
  restoring gradient `~0.001` ≪ the noise (`~0.49`) and the bias (`~0.045`). The ridge sized for
  the *estimator* (negligible bias) is far too weak to anchor a *free-running sampler*; a ridge
  large enough to anchor would bias the estimate. **Anchoring must come from the algorithm, not the
  ridge.**

**Result:** pooled corr(SE_bm, SE_boot) ≈ 0.22, slope ≈ 0.06 — fails. Mechanism real (d1 works),
implementation wrong.

**The fix (real build, not a tweak):**
1. **Decaying LR (true Polyak–Ruppert)**, `lr_k = lr₀/(1+k)^α`, α∈(0.5,1) — the shrinking step is
   the restoring force, so the iterate converges to `θ̂` even in flat directions; the average is
   `√n`-optimal. Constant LR has both the bias and the non-mixing problem.
2. **SA batch-means** (Zhu–Chen–Wu online estimator), not the stationary batch-means used here —
   the decaying-LR path is non-stationary.
3. **Report identified functionals** (`WWᵀ`) so the flat-direction wander is projected out.
4. **Calibrate the one scaling constant** against the bootstrap (we have it at p=50).

Validate against `results/validation.npz` (ground truth). Files: `polyak_bm.py` (constant-LR
prototype + stability diagnostics), `polyak_stability.png` (the drift plot).

---

## 11. Score-sandwich attempt + gllvm comparison — the settled conclusion

**Sandwich attempt (`sandwich.py`).** Built the deterministic bread three ways, all via the Godambe
identity `A = E[ψ·sᵀ]` (valid because the ZQE centering makes `E_{f_θ}[ψ]=0`):
(i) complete-data Poisson score on **prior** fantasy draws; (ii) closed-form **Laplace** posterior
score `s_marginal(y)=E_{N(ẑ,Σ)}[(y−λ(u))[u,1]]` with `Σ=(WᵀD W+I)⁻¹` recomputed post-fit; (iii) (ii)
averaged over fantasy. **All fail the same way** vs `validation.npz`: SE inflated **5–47×**,
corr ≈ 0.25, **cond(A) ≈ 1e3–1e7**. The bread `A` is **structurally near-singular** (weakly-identified
directions), and inverting it explodes — independent of how `s` is computed or how many draws. The
Laplace score fixed the score *variance* but not the *conditioning*.

**The decisive check (`gllvm_se_check.py`): R `gllvm`'s OWN deterministic SEs do the same thing.**
gllvm fits well (procW 0.10–0.14), and for **typical (well-identified) loadings its Laplace
observed-information SE conforms to the bootstrap** (median ratio ≈ 2×, same order). **But in the
weakly-identified directions gllvm's SE also blows up** — 90th-pct SE up to 3.7, and in aggregate the
Frobenius det-SE is **13× larger than gllvm's own empirical sampling spread** (0.55 vs 7.15). So the
gold-standard, published Laplace SE is *itself massively over-stated* exactly where the Hessian goes
near-singular.

**Conclusion (settled).** Inflation in the flat directions is **not a bug in our sandwich** — it is an
**intrinsic wall of deterministic observed-information / sandwich inference under near-non-identification**.
Every method that forms and inverts a sensitivity/Hessian (our Godambe bread, gllvm's TMB Laplace)
explodes there; the difference is only degree (ours 25×, gllvm 13×).

The **regularized parametric bootstrap is the robust method** — and this *elevates* it from "what
worked" to "the principled choice." It never forms or inverts a Hessian; it re-solves the *ridged*
estimator, so it stays calibrated everywhere (coverage 0.935) including the directions where all
deterministic methods inflate. The mild undercoverage (0.91–0.94) we worried about is small and
explainable (finite D, O(1/n) ridge bias) — and far better-behaved than any deterministic alternative
*including gllvm's*. Ranking under weak identification: **bootstrap ≫ gllvm-Laplace ≈ score-sandwich ≫
batch-means**.

Scalability remains the open cost: the bootstrap is `D·H` re-solves. Scaling deterministic inference
would require *taming the flat direction itself* (stronger structural identification, not a tiny ridge)
before any Hessian-based method is trustworthy — genuine future work, not a quick win.
Files: `sandwich.py`, `sandwich_check.png`, `gllvm_se_check.py`, `results/{sandwich,gllvm_se}.npz`.

## References
- Godambe (1960), *An optimum property of regular maximum likelihood estimation* — estimating-function info.
- Godichon-Baggioni, Lu, Portier (2024), arXiv:2401.10923 — online/recursive inverse-Hessian (scalable A⁻¹).
- Spall (2005), *Monte Carlo computation of the Fisher information matrix* — random-probe Hessian (§4.3).
- Chen, Lee, Tong, Zhang (2020, Ann. Statist.) — batch-means (Hessian-free) contrast.
- `consistency_map_encoder.md` (this repo) — IFT consistency + the local-nonsingularity assumption A uses.
