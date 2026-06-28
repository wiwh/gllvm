# Working notes — ZQE paper thinking

*Running record of ideas, arguments, and framings developed during the project.*

---

## Evaluation metric: orthogonal Procrustes distance (May 2026)

All simulation comparisons use the **relative orthogonal Procrustes error**:

$$d(W, \hat{W}) = \frac{\min_{R \in O(q)} \|W - \hat{W}R\|_F}{\|W\|_F}$$

Optimal rotation: SVD of $W^\top \hat{W} = USV^\top$, then $R^* = VU^\top$.

**Do NOT use `scipy.spatial.procrustes`.** That function (i) centres each matrix by
subtracting column means, and (ii) scales both matrices to unit Frobenius norm before
computing the distance. Both transformations are inappropriate for loading matrices:
- centering is not meaningful (columns of $W$ are not zero-mean by construction)
- norm-equalisation erases scale information: a shrunken estimate gets the same score
  as a well-calibrated one

Dividing by $\|W\|_F$ gives a dimensionless relative error while still penalising scale
bias (a method with weight decay that shrinks $\hat{W}$ toward zero will score worse).

For the paper, report this as "relative orthogonal Procrustes error" and cite the
orthogonal Procrustes problem (Schönemann 1966) rather than the full Procrustes analysis
(Gower 1975) to make clear no scaling/centering is applied.

---

## REVISED PAPER STRATEGY (May 2026)

### The vulnerability of the current draft

The current paper's "any surrogate works" framing is theoretically correct but strategically weak. A JRSS-B referee will object: *"The authors show what goes wrong with VAEs but provide no guidance on surrogate selection, and the identification condition is never verified for any non-trivial case. The contribution is largely negative."* That is a rejection.

### The fix: Gaussian proxy as the canonical, constructive instance

**The general framework works with any proxy.** This is the theoretical backbone and it should stay prominent. ZQE is consistent at $\theta^*$ regardless of proxy choice — Gaussian, Laplace, neural, anything. The centering argument requires no assumption on the proxy distribution.

The paper should then lead with the **Gaussian proxy family** as the canonical instance — not as the only option, but as the natural answer to "what proxy should I use in practice?" The Gaussian proxy is:
- *Principled*: $T(y)$ chosen to variance-stabilise connects to classical FA and the exponential family structure
- *Computationally trivial*: closed-form MAP encoder, no architecture choices, no encoder training
- *Theoretically clean*: identification proof goes through explicitly, sandwich covariance computable analytically

> *"The general ZQE framework is consistent for any proxy. We show that one canonical instance — the Gaussian proxy family — is both computationally trivial and theoretically grounded, properties that the VAE, despite its complexity, does not share."*

This answers the referee's question ("what do I use?") definitively without overclaiming. The Gaussian proxy is a natural anchor, not a restriction.

### Revised paper structure

| Section | Content |
|---|---|
| 1. Introduction | VAE bias problem → ZQE idea → Gaussian proxy as canonical instance |
| 2. Method | General ZQE estimating equation (brief, sets up notation) |
| 3. The Gaussian proxy family | Choose $T(y)$, closed-form encoder, deterministic objective, L-BFGS |
| 4. Theory | Jacobian cancellation, consistency, **verified for Poisson GLLVM** |
| 5. Experiments | Underspecification sweep, p-sweep, real data |
| 6. Extensions | Amortized Gaussian encoder, GP-GLLVM |

### On the amortized Gaussian encoder

The Gaussian proxy does NOT exclude amortized inference. You can train a neural encoder $q_\phi(z|y) = \mathcal{N}(\mu_\phi(y), \sigma^2 I)$ instead of using the analytic MAP. Same theory, same centering, same consistency guarantee. The Gaussian proxy is the *design principle*; how you compute the posterior is a second-order implementation choice. This makes the paper relevant to the deep learning community too, not just classical statisticians.

### What to remove

- **Experiment C (frozen encoder under decoder shift)**: VAE encoder is tightly coupled to $\theta$ via ELBO training. Moving $\theta$ while freezing the encoder creates an interpretational mess even for small shifts. This muddies the story. Remove from main paper. One sentence in discussion: *"When the surrogate is jointly trained with the decoder, distribution shift in the surrogate introduces additional complexity beyond the scope of the present analysis."*

### What to add

- **Verify Assumption 2 (local nonsingularity) for Poisson GLLVM**: with Gaussian proxy, Jacobian $= \mathbb{E}[\psi \cdot s^\top]$ where $\psi$ is linear in $T(y)$. Computable analytically. Provably nonsingular under mild conditions on $W$. This is the theoretical gap that JRSS-B will require.
- **Large-$p$ concentration theorem**: make the heuristic paragraph in the current draft into a proper theorem. Gaussian proxy makes this tractable.

### Venue: JRSS-B

Confirmed target. The GMM/estimating-equation framing, Jacobian cancellation result, sandwich covariance, and real-data scRNA-seq application are exactly the right mix. A paper there means the method exists and has been verified. Target submission in ~6 months once experiments 3 (wrong imputer) and 4 (PBMC real data) are done.

---

## Core design principle: choose T(y) to make the proxy Gaussian

The single most important design choice in ZQE is to pick $T(y)$ such that the proxy model is Gaussian:

$$T(y) \approx Wz + b + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, \sigma^2 I)$$

For Poisson data, $T = \log(1+y)$ or $T = \sqrt{y}$ (variance-stabilising) achieve this approximately. This is not a claim that $T(y)$ is truly Gaussian — it is a deliberate choice of $T$ so that the proxy *behaves* Gaussian for encoder and inference purposes.

**Gaussianity of the proxy unlocks everything in the framework:**

- **Closed-form encoder**: posterior $p(z | T(y))$ is Gaussian → MAP is $\hat{z} = (W^\top W + \sigma^2 I)^{-1} W^\top T(y)$, no sampling required
- **Deterministic centering term**: $m_2 = \mathbb{E}_\theta[T(Y)^\top W(W^\top W + \sigma^2 I)^{-1} W^\top T(Y)]$ is a quadratic form under the proxy — computable analytically, zero MC variance. More generally: draw a fixed bank of $K$ imputation samples $\{y^{(k)}_q\}$ once (fixed seed), compute encoder analytically for each, average. The centering term is then a deterministic function of $\theta$, refreshed as $\theta$ updates. This is exactly **multiple imputation**: fill in missing $z$'s analytically, average over imputed datasets. With $K=10$–$100$ this is essentially a deterministic estimator at negligible cost.
- **GP extension**: Kronecker structure on posterior precision → shared eigendecomposition across subjects, $O(T^3)$ once per $\theta$ step
- **Variance estimation**: linear solves only, no Hessian
- **GLS interpretation**: posterior precision $= \frac{1}{\sigma^2}W^\top W + I$ is the Fisher information of $T(y)$ about $z$ under the proxy — automatic feature weighting

**The score identity absorbs the misspecification.** Even though $T(y)$ is not truly Gaussian, the estimating equation is still consistent at $\theta^*$. Gaussianity is used for computational tractability, not for correctness. The two are completely decoupled.

This should be stated as the *opening design choice* in the paper: "we choose $T$ to induce a Gaussian proxy, which makes the encoder, centering term, and all downstream inference closed-form."

### Fully deterministic ZQE (and L-BFGS)

If you fix the random seed for **both**:
- the imputation bank $\{y^{(k)}_q\}$ (centering term $m_2$), and
- the posterior draws $z \sim q(\cdot|y)$ used in $m_1$,

then the entire objective $\mathcal{L}(\theta) = m_1(\theta) - m_2(\theta)$ is a **deterministic, smooth function of $\theta$**. This immediately enables second-order optimisers:

- **L-BFGS**: no learning rate, no scheduler, no patience, convergence in tens of steps. Practitioners will love this.
- **Newton-CG**, **trust-region**, or any quasi-Newton method — all valid.

The seeds can be refreshed occasionally as $\theta$ drifts (to avoid overfitting to a particular realisation), or held fixed throughout — both are consistent by the score identity.

**Inference closes the loop**: run L-BFGS to convergence at $\hat\theta$, then draw a fresh imputation bank (or reuse the fixed one) to compute the sandwich variance. The entire pipeline — estimation and inference — is deterministic given the seeds. No bootstrap, no MCMC, no Hessian inversion.

---

## Core framing: ZQE as SEM with right sufficient statistics

ZQE is a **structural equation model with the right sufficient statistics**. The estimating equations are:

$$\mathbb{E}[T(y) \cdot \eta(z)] = \mathbb{E}_\theta[T(Y) \cdot \eta(z)]$$

This is a method-of-moments condition — not a variational bound, not a KL, not a reparametrisation. A statistician can read it without knowing anything about variational inference.

- Pick $T(y)$ from the exponential family structure of the model
- Compute the ridge encoder from the proxy model in closed form
- Gradient descent on $\theta$ only

---

## Advantages of the method

### Simplicity
One low-dimensional, interpretable, domain-driven choice ($T(y)$), not a black-box architecture search. For count data the candidate list is essentially: $y$, $\log(1+y)$, $\sqrt{y}$. Domain experts can read and validate this choice.

### Scalability
No encoder parameters to optimise. Gradient steps touch only $\theta$ (the decoder). Memory and compute scale with $p$ only through the forward pass of the GLM — which is linear. The **p-sweep experiment** (N=1000, NL=10, p ∈ {10, 100, 1000, 2000}) confirms this dramatically: ZQE Procrustes error stays flat ~0.02–0.03 across the full range, while the VAE collapses to 0.63 at p=2000.

### Computability / stability
The ZQE loss is unbiased for fixed $\theta$ via the score identity — so it does not suffer from the VAE pathology where a bad encoder actively misleads $\theta$ updates. The optimisation surface is cleaner. No encoder–decoder coupling means no posterior collapse, no KL balancing, no $\beta$-VAE tuning.

### Known inference (the underrated advantage)
Once $\hat\theta$ is converged, you have a **fully specified GLM**. Plugging in the MAP estimate:

$$\hat{z}_i = (W'W + \sigma^2 I)^{-1} W' T(y_i)$$

gives:
- Point estimates of latent scores
- Analytic posterior uncertainty (the Gaussian covariance $\Sigma = \sigma^2(W'W + \sigma^2 I)^{-1}$ is closed form)
- Standard GLM machinery for confidence intervals on loadings, likelihood ratio tests, Wald tests on whether species $j$ loads on factor $k$

A VAE gives a black-box approximate posterior you cannot easily interrogate. ZQE gives a model you can hand to a biologist.

### Just-identified, no architectural choices
One $T(y)$ from the exponential family, one encoder formula. No hidden sizes, activations, depth, dropout. The only specification is the GLM family, which the domain already tells you.

### Confirmatory / structured loadings are trivial (and hard for competitors)

In many ecological and genomic applications the researcher has **prior knowledge about which loadings are zero** — e.g. species in group A do not respond to factor 2. This is the *confirmatory* GLLVM setting, as opposed to exploratory.

In ZQE, imposing known zeros is a one-line change to the decoder:

```python
# zero those entries at init, then mask their gradients permanently
g.wz.data.copy_(torch.tril(g.wz.data))          # e.g. lower-triangular anchor
g.wz.register_hook(lambda grad: grad * mask)     # mask = tril(ones), never projected
```

Adam's moment buffers for the masked entries remain zero throughout — the constraint is structural, not a post-hoc projection, and does not interfere with the optimiser. The same mechanism works for **any** sparsity pattern: block-diagonal, cross-loadings fixed to zero, sign constraints, etc.

**R `gllvm` cannot do this**. The package hardcodes an upper-triangular zero constraint on the first `q×q` block for identifiability (not user-controllable), and there is no API to impose arbitrary user-specified zero patterns on loadings. Confirmatory analysis in R requires a completely different package (e.g. `lavaan`, `blavaan`) and does not support non-Gaussian responses.

This is a genuine differentiator: ZQE handles confirmatory GLLVMs with arbitrary response families and arbitrary zero patterns, with no re-derivation and no new software.

### L1 regularisation on loadings is trivial (and non-trivial for competitors)

Because ZQE reduces to minimising a differentiable scalar objective with Adam, L1 (lasso) penalties on the loadings are a one-line addition:

```python
loss = -(m1 - m2) + lambda_l1 * g.wz.abs().sum()
```

No re-derivation, no proxy architecture changes, no variational bound to modify. This is **not** the case for VA/Laplace methods (R `gllvm`): adding a non-smooth penalty to a Laplace or variational lower bound requires either a proximal step or a smoothing approximation, and neither is exposed cleanly in the existing software.

**Consistency caveat**: L1 shrinks loadings toward zero so the fixed point of the ZQE estimating equations is no longer the MLE — same trade-off as lasso regression, biased but interpretable. For **pattern discovery** (which loadings are truly zero?) this is exactly what you want. The bias is the feature.

**TODO / think about**: 
- Is there a natural ZQE-specific choice of $\lambda$ (e.g. scaled by $\|W\|_F$ or by $p$)?
- Does the score-identity consistency argument extend to penalised ZQE in any sense (oracle property analogue)?
- Worth a sentence in the paper: *"Because ZQE reduces to minimising a differentiable scalar objective, standard regularisers such as L1 penalties on the loadings are trivially composable — a feature not shared by methods based on variational lower bounds or Laplace approximations."*
- Check whether R `gllvm` has a proper lasso-loadings mode or only ridge-type shrinkage (`lv.formula` with random effects).

### Group lasso for automatic rank/dimensionality selection

A natural extension: apply a **group lasso** penalty where each group is one column of $W$ (i.e. all $p$ loadings for latent factor $k$):

$$\text{loss} = -(m_1 - m_2) + \lambda \sum_{k=1}^q \|W_{\cdot k}\|_2$$

As $\lambda$ increases, entire columns are driven to exactly zero — factor $k$ is eliminated. The number of non-zero columns at convergence gives an automatic estimate of the true rank $q^*$. No cross-validation of $q$, no information criterion, no restarting from scratch with different $q$.

**Proximal operator** (exact for group lasso — Yuan & Lin 2006, JRSS-B; full treatment in Bach et al. 2012 "Optimization with Sparsity-Inducing Penalties", F&T ML):

$$W_{\cdot k} \leftarrow \max\!\left(0,\, 1 - \frac{\alpha\lambda}{\|W_{\cdot k}\|_2}\right) W_{\cdot k}$$

This is block soft-thresholding and is the *exact* proximal operator for the $\ell_2$ group norm — not an approximation.

**Combining with Adam is theoretically messy.** Proximal gradient is clean when the gradient step and proximal step share the same step size $\alpha$. Adam uses per-coordinate adaptive step sizes, which breaks the pairing — the threshold $\alpha\lambda$ should be per-coordinate but group lasso acts on whole columns. In practice "Adam then proximal" (ProxAdam / AdamW-style decoupled regularisation, Loshchilov & Hutter 2019 ICLR) works well empirically but convergence guarantees are weaker.

**SGD + Robbins-Monro is the theoretically clean combination.** With step size $\alpha_t = \alpha_0/t^\gamma$ (Robbins-Monro), the gradient step and proximal threshold use the *same* $\alpha_t$:

$$W_{\cdot k} \leftarrow \max\!\left(0,\, 1 - \frac{\alpha_t \lambda}{\|W_{\cdot k}\|_2}\right) W_{\cdot k}$$

No mismatch, no per-coordinate adaptation issue. This is **composite prox-SGD** (Duchi & Singer 2009; Rakhlin, Shamir & Sridharan 2012) and convergence at rate $O(1/\sqrt{T})$ is guaranteed under standard conditions. Since ZQE with the Gaussian proxy already works with SGD + Robbins-Monro step sizes (no Adam needed), lasso and group lasso slot in with zero additional machinery.

For plain entry-wise L1 on $W$, the update is even simpler — scalar soft-threshold with the same $\alpha_t$:
$$w_{jk} \leftarrow \mathrm{sign}(w_{jk})\,\max(0,\, |w_{jk}| - \alpha_t\lambda)$$

Clean theoretical options:
- **(a) FISTA** (Beck & Teboulle 2009, SIAM J. Imaging): optimal proximal gradient, step size from Lipschitz constant of $\nabla f$. Theoretically tight, no momentum. Works if ZQE is deterministic (fixed imputation bank → smooth $f$).
- **(b) Adam + proximal (AdamW-style)**: apply block soft-threshold after each Adam step. Standard in practice, the right citation is Loshchilov & Hutter (2019) for the philosophy.
- **(c) Proximal coordinate descent**: Yuan & Lin (2006) original. Exact, no step-size tuning, but sequential by column → slow on GPU.

For experiments: use (b). For any theorem: use (a).

**Regularisation path**: fit over a grid of $\lambda$ values (warm-starting from the previous solution), plot number of active factors vs $\lambda$, pick the elbow. This is the group-lasso analogue of the lasso solution path for variable selection.

**Connection to literature**: Bhattacharya & Dunson (2011) achieve the same thing via a multiplicative gamma process prior + MCMC — beautiful but slow. ZQE + group lasso is the frequentist, gradient-descent version of the same idea.

**TODO / think about**:
- Initialise with $q$ slightly too large (overfit), let group lasso prune — how sensitive is the result to the initial $q$?
- Is the proximal step correct when Adam has momentum? (Yes, proximal-gradient applies to the composite objective; Adam momentum applies only to the smooth part.)
- Worth an experiment: group-lasso path on the sparse-loadings simulation — does it recover the true $q^*$ = ACT?
- Check whether R `gllvm` has any rank-selection mechanism beyond BIC/AIC grid search over $q$.

---

## The encoder is already doing GLS (not OLS)

Under the Gaussian proxy, the posterior precision is:

$$\Sigma^{-1} = \frac{1}{\sigma^2} W'W + I$$

When $y_j$ is highly informative about $z$ (large $|W_{j\cdot}|$), that feature pulls the posterior mean strongly and sharpens $\Sigma^{-1}$. Features with small loadings contribute little to $W'W$ and barely move the posterior — they are naturally downweighted. **The weighting is automatic and implicit.**

The precision matrix of the posterior *is* the Fisher information of $T(y)$ about $z$ under the proxy. So the encoder is automatically weighting by signal strength — this is a clean way to explain why the encoder is principled rather than ad hoc.

A peaked posterior = highly informative $y$ = the encoder is confident. A flat posterior = weakly informative $y$ = the encoder diffuses. No explicit feature weighting is needed; the posterior curvature captures it.

---

## The score identity: why encoder misspecification doesn't matter

The key theoretical result is that the score identity:

$$\mathbb{E}_\theta[\nabla_\theta \log p_\theta(Y)] = 0$$

means that any bias introduced by using a misspecified encoder $q(z|y)$ instead of the true posterior $p_\theta(z|y)$ vanishes at $\theta^*$. This is why:

1. **No encoder training is required** — the ridge encoder is fixed from step one and never updated
2. **Misspecification of the proxy is harmless for $\theta$ consistency** — even if $\log(1+y)$ is not the true sufficient statistic, the estimator is still consistent
3. **Blocking the gradient through the encoder** (using `torch.no_grad()` on encoder samples) is correct — the encoder bias cancels exactly at $\theta^*$ via the score identity; blocking removes only noise-inducing chain-rule terms

---

## On specifying $T(y)$: an honest account

Users must specify $T(y)$. This is a real choice and should be acknowledged in the paper. However:

- It is a **much smaller choice** than an encoder architecture
- The exponential family gives a natural default: for Poisson, the canonical sufficient statistic is $y$ itself; $\log(1+y)$ is a deliberate proxy misspecification that stabilises variance
- It is **interpretable and checkable**: a domain expert can validate it; a broken $T(y)$ manifests as a ZQE loss that won't converge cleanly
- The VAE also implicitly specifies $T(y)$ — baked into the reconstruction loss — but users don't think of it that way. ZQE makes the choice explicit and first-class, which forces the analyst to think about what summary of $y$ is informative about $z$. This is a feature, not a bug.

---

## Experimental results summary (simulation_sparse/exp_A_sparse_loadings.ipynb)

**Fixed-p comparison** (NL=10, ACT=10, NR=100, N=1000, seed=42):

| Method | Procrustes |
|---|---|
| VAE (staged lr, trained encoder) | 0.0296 |
| MAP log1p(y) — no encoder training | **0.0242** |
| Gauss log1p(y) — no encoder training | 0.0247 |

ZQE beats a carefully tuned VAE (warmup + 4 lr-decay stages + early stopping) with zero encoder training. The VAE had every advantage in this regime (all 10 latents active, moderate p).

**p-sweep** (NL=10, ACT=10, N=1000, seed=42, p ∈ {10, 100, 1000, 2000}):

| p | VAE | MAP log1p | Gauss log1p |
|---|---|---|---|
| 10 | 0.055 | ~0.030 | ~0.030 |
| 100 | 0.030 | ~0.024 | ~0.025 |
| 1000 | 0.082 | ~0.025 | ~0.025 |
| 2000 | **0.630** | ~0.025 | ~0.025 |

ZQE is **invariant to p**. The VAE collapses at high p because its fixed-capacity encoder cannot map $\mathbb{R}^p \to \mathbb{R}^k$ reliably. ZQE's ridge encoder gets *more* accurate as p grows (more measurements per latent → ridge regression concentrates), so the ZQE advantage grows with p. This is the opposite of what you'd expect from a "simpler" method.

---

## Things to not include in the main story (decisions made)

- **GMM / overidentified systems**: adding multiple $T(y)$ dilutes the message. Stay just-identified (one $T(y)$).
- **Deep neural net amortized encoder**: also dilutes the message. The whole point is that no encoder training is needed.
- **Whitening the encoder**: whitening is absorbed by the linear $\eta(z) = Wz$ in the ZQE loss — no improvement on $\theta$. Not worth mentioning.

---

## ZQE as a general estimating equation framework

The method-of-moments framing reveals something broader: **ZQE is a recipe for constructing estimating equations for any latent variable model.** The condition

$$\mathbb{E}[T(y) \cdot \eta(z_{\rm enc}(y))] = \mathbb{E}_\theta[T(Y) \cdot \eta(z_{\rm enc}(Y))]$$

is an estimating equation in $\theta$. The user supplies:
1. $T(y)$ — a summary of the data
2. $z_{\rm enc}(y)$ — any encoder (MAP, posterior mean, posterior sample, random projection, …)

Neither needs to be exact. The score identity guarantees consistency at $\theta^*$ regardless of misspecification in both. The analyst is not choosing an approximation to the ELBO; they are choosing which **moment** to match. This reframes the method as a principled alternative to GMM for latent variable models where GMM moments are not available in closed form.

### The moment is a nested expectation — MAP is the degenerate case

The precise ZQE moment is:

$$g(y, \theta) = \mathbb{E}_{z \sim q(z|y,\theta)}[T(y) \cdot \eta(z)] - \mathbb{E}_{Y \sim p_\theta}\mathbb{E}_{z \sim q(z|Y,\theta)}[T(Y) \cdot \eta(z)]$$

This is a **doubly-nested expectation** — the moment itself requires integrating out $z$ under the current encoder $q(z|y,\theta)$. For the MAP encoder, $q$ is a delta mass so this collapses to $T(y)\cdot\eta(\hat{z}_{\rm MAP}(y,\theta))$ — no inner stochasticity. For the Gaussian posterior encoder, a single sample $z \sim q(\cdot|y,\theta)$ gives an unbiased MC estimate of the inner expectation. Both are estimable by the same SGD routine: sample $z \sim q(\cdot|y)$, compute $T(y)\cdot\eta(z)$, backprop through $\eta(z)$ only (encoder under `no_grad`). The doubly-stochastic structure (over $y$ and over $z|y$) adds variance but no bias.

---

### Subsampling observations and responses is free

Nothing forces $T(y)$ to use all $p$ responses, or all $T$ time points in the longitudinal case. You can:

- Use a **random subset of $p' \ll p$ responses** per step (feature subsampling). The moment condition holds in expectation; subsampling gives an unbiased gradient with lower memory cost.
- For the **centering term** $\mathbb{E}_\theta[T(Y) \cdot \eta(z_{\rm enc}(Y))]$, take a manageable Monte Carlo approximation — again, unbiased.

This means for very high-$p$ settings you can trade variance for memory without any theoretical cost. The ZQE scalability advantage shown in the p-sweep is therefore a *lower bound* on what is achievable — you can always subsample responses to keep cost constant in $p$.

---

## Future directions (noted, not for now)

- **Weighted $T(y)$**: could weight features by their expected contribution to the likelihood (heteroscedastic correction for proxy misspecification). But the posterior curvature already does this implicitly via $W'W$, so the marginal gain is likely small.
- **N-sweep**: confirm ZQE advantage grows as N→∞ (consistency rate)
- **Multiple seeds**: robustness check on p-sweep results

---

## Post-hoc variance estimation (almost free standard errors)

Once $\hat\theta$ is converged, standard errors on $\hat\theta$ are available via a cheap Monte Carlo procedure — no Fisher information matrix, no Hessian, no bootstrap of the full optimisation.

**Procedure:**
1. Fix $\hat\theta$, fix encoder. For $h = 1,\ldots,H$ (say $H=1000$):
   - Draw $Y_q^{(h)} \sim p_{\hat\theta}$ (ancestral sampling — one forward pass)
   - Solve $\hat{z}^{(h)} = (W^\top W + \sigma^2 I)^{-1} W^\top T(Y_q^{(h)})$ — one linear solve, closed form
   - Store $s^{(h)} = T(Y_q^{(h)}) \cdot \eta(\hat{z}^{(h)})$
2. $\widehat{\text{Var}}(\hat m_2) = \frac{1}{2} \cdot \text{empirical variance of } \{s^{(h)}\}$

The factor $1/2$: each $s^{(h)}$ is a single-draw estimator of $m_2$, which has $2\times$ the variance of the two-draw estimator used during training. So dividing by 2 corrects for this.

From $\widehat{\text{Var}}(\hat m_2)$, standard errors on $\hat\theta$ follow via the sandwich estimator for estimating equations:

$$\text{Var}(\hat\theta) \approx J^{-1} \widehat{\text{Var}}(g) (J^{-1})^\top$$

where $J = \partial g / \partial \theta$ at $\hat\theta$ (computable by one autodiff pass). Alternatively, run the procedure for small perturbations of $\hat\theta$ to estimate $J$ numerically.

**Common random numbers (CRN)**: use the same latent seeds $z_0^{(h)}$ for drawing $Y_q^{(h)}$ across all replicates, varying only Poisson noise. Induces positive correlation between $m_1$ and $m_2$ (same latent structure), reducing $\text{Var}(m_1 - m_2)$ below $\text{Var}(m_1) + \text{Var}(m_2)$. Fewer replicates needed for the same precision.

**Why this is special**: standard GLLVMs get standard errors from the observed Fisher information — $O(p^2 q^2)$ to compute and invert. This procedure costs $H$ forward passes + $H$ linear solves. Scales to large $p$ trivially, and requires zero modification of the training code.

---

## Identification theory for the Gaussian-proxy ZQE estimator

*This is the main theoretical contribution needed for JRSS-B. The proofs below are essentially complete. The b-block lemma is clean and ready to write up. The W-block goes through a large-p concentration argument that upgrades to a proper theorem. An IFT remark handles gradient computation for non-Gaussian proxies.*

---

### Setting and notation

Consider a GLLVM where responses $Y_1,\ldots,Y_p$ are **conditionally independent** given $Z \in \mathbb{R}^q$, each from a one-parameter exponential family:
$$f_j(y_j | z;\theta) = \exp\bigl\{\tilde{T}_j(y_j)\,\eta_j(z;\theta) - A_j(\eta_j(z;\theta)) + B_j(y_j)\bigr\}$$
with natural parameter $\eta_j(z;\theta) = w_j^\top z + b_j$, prior $Z \sim \mathcal{N}(0, I_q)$, and $\theta = (W, b)$ with $W = [w_1|\cdots|w_p]^\top \in \mathbb{R}^{p\times q}$.

**The Gaussian proxy.** Choose a transformation $T_j : \mathbb{R} \to \mathbb{R}$ (e.g. $\log(1+y_j)$ for Poisson, $\arcsin\sqrt{y_j/n}$ for Binomial) and *pretend* that
$$T_j(Y_j) | Z \sim \mathcal{N}(w_j^\top Z + b_j,\, \sigma^2), \quad \text{independently.}$$
This is deliberately false. It is a computational device. The MAP encoder under this proxy is:
$$\hat{z}(y;\theta) = \underbrace{(W^\top W + \sigma^2 I)^{-1}}_{\Sigma_z} W^\top (T(y) - b)$$
where $T(y) = (T_1(y_1),\ldots,T_p(y_p))^\top$. The estimating function is:
$$\psi(\theta; y) = \begin{pmatrix} \psi_W(\theta;y) \\ \psi_b(\theta;y) \end{pmatrix}, \quad \psi_b(\theta;y) = \hat{z}(y;\theta) \otimes T(y), \quad \psi_W(\theta;y) = \hat{z}(y;\theta) \otimes \hat{z}(y;\theta)^\top W \cdots$$

*(cleaner: use the natural parametrisation from the main text. The key objects are $D(z;\theta) = \partial\eta/\partial\theta$ and $T(y)$ as per the general framework. Concretely: for the bias block, $\psi_{b_j}(\theta;y) = \hat{z}_k$-weighted $T_j(y_j)$ terms. See main text eq.)*

The Jacobian is $A(\theta^*) = \mathbb{E}[\psi(\theta^*;Y)\,s(\theta^*;Y)^\top]$ where $s(\theta^*;Y)$ is the true marginal score.

The true score decomposes response-by-response (using the exponential family complete-data score and conditional independence):
$$s_{b_j}(\theta^*;y) = \mathbb{E}_{Z|Y=y}\bigl[\tilde{T}_j(Y_j) - \tilde{\mu}_j(Z)\bigr] = \tilde{T}_j(y_j) - \mathbb{E}_{\theta^*}[\tilde{\mu}_j(Z) | Y=y]$$
where $\tilde{\mu}_j(z) = A_j'(\eta_j(z;\theta^*)) = \mathbb{E}[\tilde{T}_j(Y_j)|Z=z]$ is the true conditional mean.

---

### Lemma 1 (b-block): $A_{bb}$ is diagonal and positive definite

**Statement.** For all $j \neq k$: $(A_{bb})_{jk} = 0$. For each $j$:
$$(A_{bb})_{jj} = \mathbb{E}\bigl[\operatorname{Cov}(T_j(Y_j),\, \tilde{T}_j(Y_j) \mid Z)\bigr] > 0.$$

**Proof.**

Fix $j, k$. By the Jacobian identity (eq. from main text), $(A_{bb})_{jk} = \operatorname{Cov}(T_j(Y_j),\, s_{b_k}(\theta^*;Y))$.

Write the true score residual as:
$$s_{b_k}(\theta^*;Y) = \underbrace{\tilde{T}_k(Y_k) - \tilde{\mu}_k(Z)}_{\text{(i) noise residual}} + \underbrace{\tilde{\mu}_k(Z) - \mathbb{E}[\tilde{\mu}_k(Z)|Y]}_{\text{(ii) posterior residual}}$$

**Term (i).** By the law of total covariance:
$$\operatorname{Cov}(T_j(Y_j),\, \tilde{T}_k(Y_k) - \tilde{\mu}_k(Z)) = \mathbb{E}[\operatorname{Cov}(T_j(Y_j),\, \tilde{T}_k(Y_k) \mid Z)]$$
since $\mathbb{E}[\tilde{T}_k(Y_k) - \tilde{\mu}_k(Z) \mid Z] = 0$ kills the between-$Z$ term.  
By **conditional independence** ($Y_j \perp Y_k \mid Z$ for $j \neq k$): this equals zero when $j \neq k$, and $\mathbb{E}[\operatorname{Cov}(T_j(Y_j), \tilde{T}_j(Y_j) \mid Z)]$ when $j = k$.

**Term (ii).** The posterior residual $\tilde{\mu}_k(Z) - \mathbb{E}[\tilde{\mu}_k(Z)|Y]$ is orthogonal (in $L^2$) to every $\sigma(Y)$-measurable function, by the definition of conditional expectation. Since $T_j(Y_j) \in \sigma(Y)$, this covariance is zero.

**Positivity.** The diagonal entry $\mathbb{E}[\operatorname{Cov}(T_j(Y_j), \tilde{T}_j(Y_j) \mid Z)]$ is positive whenever:
- $T_j$ is strictly increasing, and
- $Y_j \mid Z$ has positive conditional variance (guaranteed for any exponential family with non-degenerate dispersion).

Under these conditions, $T_j(Y_j)$ and $\tilde{T}_j(Y_j)$ are strictly positively conditionally correlated given $Z$. $\square$

**Corollaries.** The result applies simultaneously to all standard exponential families:
- **Poisson**, $T_j = \log(1+y_j)$, $\tilde{T}_j = y_j$: $\operatorname{Cov}(\log(1+Y_j), Y_j | Z) > 0$ since $Y_j|Z \sim \text{Poisson}(\lambda_j(Z))$.
- **Poisson**, $T_j = \sqrt{y_j}$ (Anscombe): same reasoning.
- **Binomial**, $T_j = \arcsin\sqrt{y_j/n}$: variance-stabilising, strictly increasing, positive conditional variance.
- **Gamma**, $T_j = \log y_j$: $Y_j|Z \sim \text{Gamma}$, log is strictly increasing.
- **Negative Binomial**: any strictly increasing $T_j$ works.

---

### Lemma 2 (W-block, large-p): $A_{WW}$ is nonsingular for all $p$ large enough

**Assumptions for this lemma.**

Let $p \to \infty$ with $q$ fixed. Write $W_p = [w_1|\cdots|w_p]^\top \in \mathbb{R}^{p \times q}$. Assume:

- **(A1) Stable spectrum**: $\lambda_{\min}(W_p^\top W_p / p) \geq c > 0$ for all $p$ large enough, for some constant $c$ depending only on $\theta^*$.
- **(A2) Bounded leverage**: $\max_{1 \leq j \leq p} \|w_j\|^2 / p \to 0$ as $p \to \infty$.
- **(A3) Finite conditional variance**: $\sup_j \operatorname{Var}(T_j(Y_j) | Z) \leq \kappa < \infty$ almost surely, uniformly in $j$ and $p$.
- **(A4) Linearisation bias bounded**: $\sup_j |\mathbb{E}[T_j(Y_j)|Z] - (w_j^\top Z + b_j)| \leq C\|w_j\|$ for some constant $C$ depending only on the exponential family. (For Poisson with $T_j = \log(1+y_j)$ this follows from $\mathbb{E}[\log(1+Y_j)|Z] \approx \log(1+\lambda_j(Z)) \approx w_j^\top Z + b_j$ when $\lambda_j$ is small to moderate.)

*Remark:* (A1) says the loading matrix is asymptotically of full rank in the sense of random matrix theory — satisfied trivially for fixed $W$ with a bounded-below smallest singular value, or for random loadings with $\mathbb{E}[w_j w_j^\top] \succ 0$. (A2) is a standard Lindeberg condition ruling out a single dominant row. Both are mild.

---

**Step 1: Encoder concentration.**

Write the true conditional mean under $\theta^*$ as $\mu_j(Z) := \mathbb{E}[T_j(Y_j) | Z]$. Decompose:
$$T_j(Y_j) - b_j = \underbrace{(w_j^\top Z)}_{\text{signal}} + \underbrace{(\mu_j(Z) - b_j - w_j^\top Z)}_{\text{bias}\;:= \beta_j(Z)} + \underbrace{(T_j(Y_j) - \mu_j(Z))}_{\text{noise}\;:= \xi_j(Y_j,Z)}$$

so that $T(Y) - b = WZ + \beta(Z) + \xi(Y,Z)$ in vector form. The MAP encoder is:
$$\hat{z}(Y;\theta^*) = \Sigma_z W^\top(T(Y) - b) = \underbrace{\Sigma_z W^\top W}_{\to I} Z + \Sigma_z W^\top \beta(Z) + \Sigma_z W^\top \xi(Y,Z)$$
with $\Sigma_z = (W^\top W + \sigma^2 I)^{-1}$.

Write $\hat{z}(Y;\theta^*) = Z + \delta_p(Y,Z)$ where:
$$\delta_p := -\sigma^2 \Sigma_z Z + \Sigma_z W^\top \beta(Z) + \Sigma_z W^\top \xi(Y,Z).$$

**Bounding each term in $\mathbb{E}[\|\delta_p\|^2]$:**

*(i) Shrinkage term:* $\|-\sigma^2 \Sigma_z Z\|^2 \leq \sigma^4 \|\Sigma_z\|_{\rm op}^2 \|Z\|^2$.  
Under (A1), $\lambda_{\min}(W^\top W) \geq cp$, so $\|\Sigma_z\|_{\rm op} \leq 1/(cp)$. Thus:
$$\mathbb{E}\bigl[\sigma^4\|\Sigma_z\|_{\rm op}^2\|Z\|^2\bigr] \leq \frac{\sigma^4 q}{c^2 p^2} = O(p^{-2}).$$

*(ii) Bias term:* $\|\Sigma_z W^\top \beta(Z)\|^2 \leq \|\Sigma_z\|_{\rm op}^2 \|W^\top \beta(Z)\|^2$.  
Under (A4), $|\beta_j(Z)| \leq C\|w_j\|$, so $\|W^\top \beta(Z)\|^2 \leq C^2 \sum_k (\sum_j w_{jk}\|w_j\|)^2 \leq C^2 \|W\|_F^2 \|W\|_F^2 / p \cdot p = C^2 \|W\|_F^4 / p$ (by Cauchy--Schwarz). Using $\|W\|_F^2 = O(p)$ (since $\mathbb{E}[\|w_j\|^2]$ is bounded), this is $O(p)$. Hence:
$$\mathbb{E}\bigl[\|\Sigma_z W^\top \beta(Z)\|^2\bigr] \leq \frac{1}{c^2p^2} \cdot O(p) = O(p^{-1}).$$

*(iii) Noise term:* The $\xi_j$ are conditionally independent with mean zero given $Z$. Thus $\mathbb{E}[W^\top \xi \xi^\top W \mid Z] = \sum_j \operatorname{Var}(T_j(Y_j)|Z) w_j w_j^\top \leq \kappa W^\top W$ entry-wise by (A3). Therefore:
$$\mathbb{E}\bigl[\|\Sigma_z W^\top \xi\|^2\bigr] = \mathbb{E}\bigl[\operatorname{tr}(\Sigma_z W^\top \xi \xi^\top W \Sigma_z)\bigr] \leq \kappa \operatorname{tr}(\Sigma_z W^\top W \Sigma_z) \leq \kappa \operatorname{tr}(\Sigma_z) \leq \frac{\kappa q}{cp}.$$

Combining all three terms:
$$\mathbb{E}\bigl[\|\delta_p(Y,Z)\|^2\bigr] \leq \frac{\kappa q}{cp} + O(p^{-1}) = O(p^{-1}),$$
so $\delta_p = O_{\rm L_2}(p^{-1/2})$, i.e. $\hat{z}(Y;\theta^*) = Z + O_{\rm L_2}(p^{-1/2})$.

---

**Step 2: Remainder bound on the Jacobian.**

Write $\psi(\theta^*;Y) = s(\theta^*;Y) + \rho_p(Y,Z)$ where $\rho_p$ captures the difference between the proxy estimating function and the true score. Explicitly, both $\psi$ and $s$ are linear in $\hat{z}$ and $Z$ respectively (via the exponential family structure), so $\|\rho_p\| \leq L\|\delta_p\|$ for a Lipschitz constant $L$ depending on $\theta^*$ and the exponential family.

Then:
$$A(\theta^*) = \mathbb{E}[\psi(\theta^*;Y) s(\theta^*;Y)^\top] = \underbrace{\mathbb{E}[s(\theta^*;Y) s(\theta^*;Y)^\top]}_{= \mathcal{I}(\theta^*)} + R_p$$
where $\|R_p\|_{\rm op} \leq \mathbb{E}[\|\rho_p\| \|s\|] \leq L\,\mathbb{E}[\|\delta_p\|\,\|s\|] \leq L\,\bigl(\mathbb{E}[\|\delta_p\|^2]\bigr)^{1/2}\bigl(\mathbb{E}[\|s\|^2]\bigr)^{1/2}$, by Cauchy--Schwarz.

Under standard moment conditions on the exponential family, $\mathbb{E}[\|s(\theta^*;Y)\|^2] = \operatorname{tr}(\mathcal{I}(\theta^*)) < \infty$ for each fixed $p$, and is $O(p)$ as $p$ grows (it is a sum of $p$ independent terms). Hence:
$$\|R_p\|_{\rm op} \leq L\cdot O(p^{-1/2})\cdot O(p^{1/2}) = O(1),$$
which does not suffice as stated. We need a normalised version.

**Normalised statement.** Working with the *per-feature* estimating function $\bar\psi = p^{-1/2}\psi$ and $\bar s = p^{-1/2}s$ (which have $O(1)$ norms), the remainder becomes $\|\bar R_p\|_{\rm op} = O(p^{-1/2})$. Since $A(\theta^*) = p\cdot\bar A(\theta^*)$ with $\bar A = \mathcal{I}(\theta^*)/p + \bar R_p$, nonsingularity of $\bar A$ (and hence of $A$) follows from:
$$\lambda_{\min}(\mathcal{I}(\theta^*)/p) > \|\bar R_p\|_{\rm op}$$
i.e. $\lambda_{\min}(\mathcal{I}(\theta^*)/p) \geq c' > 0$ (which holds under (A1), since the Fisher information also scales as $O(p)$) vs $\|\bar R_p\|_{\rm op} = O(p^{-1/2}) \to 0$. $\square$

---

**Step 3: Uniformity near $\theta^*$.**

The argument above is pointwise at $\theta^*$. For the implicit function theorem (used in the asymptotic normality proof) we need $A(\theta)$ to be nonsingular in a neighbourhood of $\theta^*$.

Since $A(\theta) = \mathbb{E}[\psi(\theta;Y) s(\theta^*;Y)^\top]$ and $\psi(\theta;Y)$ is continuous in $\theta$ (the MAP encoder and GLM log-likelihood are both smooth in $(W,b)$), dominated convergence gives continuity of $\theta \mapsto A(\theta)$ in operator norm, provided $\|\psi(\theta;Y)\| \leq g(Y)$ with $\mathbb{E}[g(Y)^2] < \infty$ in a ball $\|\theta - \theta^*\| \leq \varepsilon$.

Under (A3) and the integrability of the GLM, this domination holds for any $\varepsilon < \varepsilon_0(\theta^*)$. Since nonsingularity is an open condition in operator norm, $A(\theta)$ is nonsingular in a neighbourhood of $\theta^*$ for all $p$ large enough. $\square$

---

### Proposition (Local identification)

**Statement.** Consider an exponential-family GLLVM with conditionally independent responses and $Z \sim \mathcal{N}(0,I_q)$. Let the proxy be Gaussian with $T_j$ strictly increasing for each $j$. Then:

1. **(b-block, all $p$)** $A_{bb} \succ 0$. No condition on $p$ or $W$.
2. **(Full Jacobian, large $p$)** $A(\theta^*)$ is nonsingular for all $p$ sufficiently large, provided $W$ has full column rank.
3. **(Perturbation sufficient condition)** $A(\theta^*)$ is nonsingular whenever $\lambda_{\min}(\mathcal{I}(\theta^*)) > \|A(\theta^*) - \mathcal{I}(\theta^*)\|_{\rm op}$, which is verifiable numerically at any given $\theta^*$.

**Consequence.** Under these conditions, the Gaussian-proxy ZQE estimator is consistent and asymptotically normal at $\theta^*$, with sandwich covariance $\Sigma = A^{-1} B A^{-\top}$.

---

### Remark: implicit function theorem for encoder Jacobians

During optimisation (not identification), one sometimes needs $\nabla_\theta \psi(\theta;y) = \nabla_\theta [\hat{z}(y;\theta)^\top D(\cdot;\theta)^\top T(y)]$, which involves $\partial \hat{z}/\partial \theta$.

For the MAP encoder, $\hat{z}$ solves the proxy posterior FOC: $G(\hat{z}, \theta, y) := W^\top(T(y)-b) - (W^\top W + \sigma^2 I)\hat{z} = 0$. By the implicit function theorem:
$$\frac{\partial \hat{z}}{\partial \theta} = -\left(\frac{\partial G}{\partial z}\right)^{-1} \frac{\partial G}{\partial \theta} = \Sigma_z \frac{\partial}{\partial \theta}[W^\top(T(y)-b)]$$
(since $\partial G/\partial z = -(W^\top W + \sigma^2 I) = -\Sigma_z^{-1}$). For the Gaussian proxy this is just the Jacobian of the closed-form formula, so IFT adds no new information here. However, for a **non-Gaussian proxy** (e.g. Laplace approximation, or a Newton-step encoder), where $\hat{z}$ is the output of an iterative solver, the IFT gives $\partial\hat{z}/\partial\theta$ *without* unrolling the solver — useful for L-BFGS with implicit encoders.

**The crucial point for theory**: even though $\partial\hat{z}/\partial\theta \neq 0$ in general, the Jacobian cancellation identity shows these derivatives *do not appear* in $A(\theta^*)$. So the IFT is an optimisation tool, not a proof ingredient. The consistency proof is free of encoder derivatives.

---

---

## Scalability: the three regimes where VA/LA break down (May 2026)

The "ZQE loses a little efficiency on simple cases (q=1, moderate p, iid prior)" framing is the *correct* framing. The flip side is the argument that silences objections:

| Regime | VA/LA | ZQE |
|---|---|---|
| Large $q$ | ✗ intractable integral ($q \gtrsim 5$) | ✓ MC draw from prior |
| GP prior | ✗ $n \times n$ covariance, LA needs $nq \times nq$ Hessian | ✓ sample from GP (Cholesky once) |
| Large $p$ | ✗ cannot sub-sample features without breaking the bound | ✓ sub-sample both $n$ and $p$ |
| Large $n$ | ✗ (LA especially) | ✓ mini-batch |
| Mixed response types | ✗ single `family=` string, all columns share one distribution | ✓ per-column `GLMFamily`, same loss |

**The $p$-subsampling argument is specific to ZQE.** The ZQE objective is a difference of averages over $j$:

$$\hat{\mathcal{L}} = \frac{1}{|\mathcal{B}_p|} \sum_{j \in \mathcal{B}_p} \left[ \log p(y_j | z) - \log p(y_j^q | z^q) \right]$$

Under the null, the two terms have the same distribution, so the estimator is unbiased regardless of which features are sampled. This is *not* an approximation — the fixed point of the subsampled estimating equation is the same $\theta^*$. Memory cost is $O(|\mathcal{B}_n| \times |\mathcal{B}_p| \times q)$, fully controllable.

**Demonstrated scale:** $p = 600\text{k}$, $q = 50$, $n = 10\text{k}$ runs fast with feature sub-sampling. VA/LA cannot even store the Hessian at this scale.

**Narrative for paper/talk:** *"On simple Poisson-GLLVM with small $q$ and moderate $p$, ZQE is competitive with VA/LA and slightly less efficient — a fair price for generality. Then switch on a GP prior, or go to $p = 600\text{k}$, or $q = 50$. VA and LA are gone. ZQE runs."*

---

## Mixed response types

R `gllvm` accepts a single `family=` argument (a string such as `"poisson"`, `"negative.binomial"`, `"gaussian"`, etc.) that applies uniformly to **all columns** of the response matrix. There is no `family=list(...)` interface; a dataset with, say, columns of count data, binary indicators, and continuous measurements cannot be modelled jointly in a single `gllvm()` call. The user would need to split the response into homogeneous blocks and fit separate models — which destroys the shared latent structure.

**ZQE is trivially mixed-type.** Each column $j$ carries its own `GLMFamily` object (Poisson, Binomial, Gaussian, …), contributing its own term $\log p_j(y_j \mid z;\theta_j)$ to the estimating function. No re-derivation is needed. The loss is still:

$$\mathcal{L}(\theta) = -\left(\mathbb{E}\!\left[\sum_j T_j(Y_j)\cdot\eta_j(\tilde{z}_{\rm enc})\right] - \mathbb{E}_\theta\!\left[\sum_j T_j(Y_j^q)\cdot\eta_j(\tilde{z}_{\rm enc}^q)\right]\right)$$

where $T_j$ and $\eta_j$ are the sufficient statistic and natural parameter of column $j$'s own exponential family. The Gaussian proxy encoder is also unchanged — it only requires the linearised mean $\tilde{\mu}_j(z) = w_j^\top z + b_j$ for each column, which is the same regardless of the exponential family. The encoder precision is $\sum_j \operatorname{Var}(T_j(Y_j)\mid z)^{-1} w_j w_j^\top$, which simply sums over columns with column-specific weights — still Gaussian, still closed-form.

**Practical consequence.** A joint ordination of species counts (Poisson), presence/absence (Binomial), and a continuous trait (Gaussian) is one `run_zqe(...)` call with `families=[Poisson(), Binomial(), Gaussian()]`. This is a routine feature of ZQE's architecture, not an extension.

---

## GP-GLLVM extension (longitudinal / temporal data)

This is an open problem in the longitudinal GLLVM literature. ZQE has a natural extension.

**Model:** For subject $i$, latent trajectory $z_i(t) \sim \mathcal{GP}(0, K_\theta)$ on a fixed grid $t_1, \ldots, t_T$. Observations $y_{ij}(t) \sim \text{Poisson}(\exp(w_j^\top z_i(t) + b_j))$.

**The key insight:** If all subjects share the same observation grid, the kernel matrix $K_\theta \in \mathbb{R}^{T \times T}$ is the same for everyone. Its eigendecomposition $K_\theta = V \Lambda V^\top$ is computed **once per $\theta$ update** — $O(T^3)$ total, amortised over $N$ subjects. This is the expensive step, but it's shared.

**The encoder:** Under a Gaussian proxy (linearised likelihood), the posterior over the full trajectory $z_i = \text{vec}(z_i(t_1), \ldots, z_i(t_T)) \in \mathbb{R}^{T \times q}$ is Gaussian with precision:

$$\Sigma_{\rm post}^{-1} = K_\theta^{-1} \otimes I_q + \frac{1}{\sigma^2} I_T \otimes W^\top W$$

Using the Kronecker structure and the shared eigendecomposition, this inverts in $O(T \cdot q^3)$ per subject (not $O((Tq)^3)$). The posterior mean and a whitened sample are then:

$$\mu_{{\rm post},i} = \Sigma_{\rm post} \cdot \frac{1}{\sigma^2}(I_T \otimes W^\top) T(y_i) \qquad \epsilon_i \sim \mathcal{N}(0, I_{T \times q}) \to \tilde{z}_i = \mu_{{\rm post},i} + \Sigma_{\rm post}^{1/2} \epsilon_i$$

**ZQE loss for GP-GLLVM:**

$$\mathcal{L}(\theta) = -\left(\mathbb{E}[T(y) \cdot \eta(\tilde{z}_{\rm enc}(y))] - \mathbb{E}_\theta[T(Y) \cdot \eta(\tilde{z}_{\rm enc}(Y))]\right)$$

where $\eta(\tilde{z})$ is the GLM linear predictor evaluated at the trajectory sample. No ELBO over trajectories. No encoder training. Kernel parameters $\theta_K$ (lengthscale, variance) are estimated via the same ZQE moment conditions — the kernel enters only through the eigendecomposition, which differentiates cleanly.

**Why this matters:** Existing longitudinal GLLVM methods either (a) use Laplace approximation per subject per step (slow, non-scalable), (b) amortise with a deep encoder over time series (architecture choices, instability), or (c) discretise and treat time as a nuisance. ZQE would give a closed-form encoder for the GP case with no architectural choices and a well-understood computational cost: $O(T^3)$ per $\theta$ step (shared eigendecomp) + $O(N \cdot T \cdot q^3)$ for posterior evaluations. For moderate $T$ (say $T \leq 100$) this is entirely tractable.

**Time subsampling is specific to ZQE.** Instead of conditioning the encoder on all $T$ time points, draw a random subset $S \subset \{1,\ldots,T\}$ of size $T' \ll T$ per step. The GP posterior conditioned on $S$ is still closed-form — it is the standard conditional Gaussian from the submatrix $K_{SS}$, computed from the already-available eigendecomposition. The moment condition with subset $S$:

$$\mathbb{E}[T(y_S) \cdot \eta(z_{\rm enc}(y_S, S))] = \mathbb{E}_\theta[T(Y_S) \cdot \eta(z_{\rm enc}(Y_S, S))]$$

still identifies the same $\theta^*$ by the score identity — you are estimating the same fixed point with more variance, not changing the target. The eigendecomposition cost drops from $O(T^3)$ to $O(T'^3)$.

This is *not* possible cleanly with a VAE: a VAE encoder trained on full trajectories cannot be re-used on subsets without retraining, and conditioning on a subset changes the ELBO target (the KL against the full GP prior is now mismatched). ZQE gets sparse-GP-style subsampling for free from the estimating equation structure, with no inducing-point parameters and no variational approximation.

---

## ZQE as a scalar objective for L-BFGS (implementation note)

The ZQE estimating equations $\Psi_n(\theta) = 0$ are not naturally an optimization problem. However, they **are** the gradient of a scalar, so L-BFGS can be used as a root-finder by minimizing that scalar.

For Poisson-GLLVM with sufficient statistic $T(y) = y$ and natural parameter $\eta(z; \theta) = Wz + b$, the scalar is:

$$
\mathcal{L}_{\mathrm{ZQE}}(\theta)
= \frac{1}{nM}\sum_{i=1}^n\sum_{m=1}^M T(y_i)^\top \eta(z_m^{(i)}, \theta)
- \frac{1}{M}\sum_{m=1}^M T(\tilde{y}_m)^\top \eta(\tilde{z}_m, \theta)
$$

where the reparameterized samples are:

$$z_m^{(i)} = \mu_q(y_i;\theta) + \sigma_q(y_i;\theta) \odot \varepsilon_m, \quad \varepsilon_m \sim \mathcal{N}(0,I) \text{ (fixed)}$$
$$\tilde{z}_m \sim f_Z \text{ (fixed)}, \quad \tilde{y}_m \sim f_{Y|Z}(\cdot \mid \tilde{z}_m; \theta) \text{ (fixed Poisson draws)}$$

**The fixed-seed trick.** Fix all noise seeds $\{\varepsilon_m\}$, $\{\tilde{z}_m\}$, $\{\tilde{y}_m\}$ once before calling L-BFGS. This makes $\mathcal{L}_{\mathrm{ZQE}}(\theta)$ a deterministic smooth function of $\theta$, so L-BFGS gets consistent gradient estimates and valid curvature history throughout the solve.

**Important:** the seed fixes the *noise* $\varepsilon_m$, not the *samples* $z_m^{(i)}(\theta)$. The reparameterization ensures $z_m^{(i)}$ still depends smoothly on $\theta$ (through $\mu_q$ and $\sigma_q$), so autograd differentiates through it correctly. Similarly, $\tilde{y}_m$ are fixed integer constants entering only through $T(\tilde{y}_m)$ — Poisson discreteness causes no gradient issue.

**L-BFGS does not compute gradients.** It reads `param.grad` filled by `.backward()` inside the closure. It terminates when $\|\nabla_\theta \mathcal{L}_{\mathrm{ZQE}}\| < \texttt{tol}$, i.e., when the estimating equations are satisfied. Feed `-`$\mathcal{L}_{\mathrm{ZQE}}$ if your solver minimizes; or maximize $\mathcal{L}_{\mathrm{ZQE}}$ directly.

**Connection to ELBO.** The ELBO is the first term of $\mathcal{L}_{\mathrm{ZQE}}$ minus a KL penalty. ZQE replaces the KL by the model-expectation centering term (second sum above). The KL keeps $q$ close to the prior; the centering term keeps $\theta$ at the correct population root regardless of $q$.

---

## GP-GLLVM identification: keep $W$ full — do NOT impose lower-triangular (2026-06-21)

For the **GP-GLLVM extension**, the loading identification is fundamentally different from the
plain GLLVM, and the standard lower-triangular constraint is **wrong** here. This is a gauge
argument.

**Plain GLLVM** ($z\sim\mathcal N(0,I_q)$): the loadings carry a full $O(q)$ rotation gauge,
$W \equiv WR$ for $R\in O(q)$ (since $Rz\stackrel{d}{=}z$). A lower-triangular constraint picks one
representative, and crucially **the truth remains in the constrained family**: exactly one rotation
lower-triangularises $W^\*$, at zero cost to the fit. The constraint is free and correct. (This is
the `wz_mask` / `tril` mechanism used in the confirmatory and sparse-loading experiments.)

**GP-GLLVM with distinct per-factor length-scales $\ell_k$:** the rotation gauge is **already
broken by the kernels**. The latent covariance is
$\operatorname{Cov}(\eta) = \sum_k w_k w_k^\top K_{\ell_k}(\cdot,\cdot)$; a rotation $R$ mixes
factors with *different* kernels and therefore *changes* this covariance — it is no longer a
symmetry. Hence $W$ is identified up to permutation and sign by the kernels themselves (this is the
classical GPFA fact, "distinct timescales break the rotation"). The consequence for the constraint:
**there is in general no rotation mapping $W^\*$ into lower-triangular form** (rotations are no
longer free), so the lower-triangular family **does not contain the truth**. Imposing it is then a
*misspecification* that biases **both** $W$ and the length-scales $\ell_k$ — the fit distorts $\ell$
to compensate for the unreachable loadings.

**Rule.** Distinct kernels $\Rightarrow$ **full unconstrained $W$** (the kernels do the
identifying; the factors *are* their length-scales). Shared/equal kernel $\Rightarrow$ the $O(q)$
gauge returns $\Rightarrow$ lower-triangular is then the right and free fix (and since the $\ell_k$
are equal there is no length-scale$\leftrightarrow$factor correspondence to break). The two
identification schemes are **alternatives, not complements**; the GP-GLLVM's whole contribution is
distinct per-factor length-scales, so it always uses a full $W$.

**On $B$ (cross-latent correlation).** Keeping $W$ full *is* keeping the freedom: an unconstrained
$W$ already carries the $W\!\cdot\!\mathrm{chol}(B)$ degree of freedom (the within-observation
cross-latent structure). We do not separately parameterise $B$ — as a distinct object it is
non-identified in the standard (shared-kernel) sense, confirmed by the GP-factor-analysis
literature — but we must not amputate it either. **Full $W$ + $B=I$ + distinct $\ell_k$** is the
identified, maximally-free model.

**Caveat.** If two factors come out with $\ell$'s that are nearly equal, those two are mutually
rotation-degenerate (a residual $O(2)$ gauge in that subspace), correctly reflected as high
variance there — *not* something to "fix" with lower-triangular (which would distort the
well-separated factors). The relative orthogonal Procrustes metric is rotation-invariant, so
loading-recovery scores are unaffected regardless.

---

## Paper framing & scope — keep the method general; GP-GLLVM is the flagship (2026-06-21)

**Decision: do NOT rename/pitch the paper around "large-scale GP-GLLVM." Keep the general
likelihood-free centered-estimation method as the headline, and *promote* the GP-GLLVM-at-scale
from a buried §Extension to a co-equal, named pillar (the flagship demonstration).**

- **Why general stays the headline (JRSS-B specifically):** B buys methodology + theory. Our ticket
  is the centered estimating-equation *principle* (Jacobian cancellation, consistency/AN under
  misspecified surrogates, identification, sandwich inference) — all of which is general across
  instances. A paper *titled* around GP-GLLVM reads as a model-specific computational method (a
  weaker B fit — that's AOAS / Biometrics / Bioinformatics turf) and throws away the theory that
  makes it a B paper.
- **Why GP-GLLVM must be elevated anyway:** "you don't need a correct encoder" is a *defensive*
  (robustness/negative) message — the referee worry. The GP-GLLVM at scale is the *offensive*
  payoff: *"…and therefore you can fit a count-valued, GP-structured latent factor model at
  $n\to$millions, where VA / Laplace / EM-GPFA cannot run."* It is the **existence proof** that the
  principle buys something no likelihood method can — it turns the negative result positive.
- **Recommended spine:** theory → Gaussian-proxy canonical instance → **GP-GLLVM at scale (the
  "reach" demo) + the Visium figure** → robustness.
- **Two-paper split (avoid cramming):**
  - *Paper 1 (JRSS-B, now):* general centered/likelihood-free decoder estimation; GLLVM sims (VAE
    bias, p-sweep, efficiency–robustness) + GP-GLLVM-at-scale flagship + one clean Visium figure +
    robustness section. Complete and *general*.
  - *Paper 2 (applied, later):* GP-GLLVM as a count-native multi-scale spatial feature extractor —
    the full "GP-PCA" vision and biology. The Visium figure teases P1, headlines P2.

## Encoder choice for the paper — Gaussian MAP canonical; quantile is surgical (2026-06-21)

**Decision: the canonical encoder is the closed-form Gaussian-MAP (log1p). Do NOT make the quantile
projection mandatory/core.** Deploy the quantile projection only where it pays: (i) the robustness
construction (calibrated $\hat\mu$ for Pearson-residual Huberization), and (ii) the GP-PCA
embeddings ($N(0,1)$ channels for downstream).

- **Reasons:** keep the identification proofs / deterministic-objective on the smooth MAP; the
  quantile's W benefit is modest (the de-shrink is a *scalar that cancels in the centered
  $m_1-m_2$ equation* — gain is variance/stability, not a 2× mean); making a non-differentiable
  batch transform mandatory invites "why this machinery?"; and it dilutes the crisp "Gaussian proxy
  is the canonical instance" pitch.
- **Empirical (Poisson GLLVM, 20 seeds, paired; `playground/poisson_quantile.ipynb`):** MAP+quantile
  is the *best* encoder for $W$ — lowest mean (0.075 vs 0.080) and ~half the variance (±0.011 vs
  ±0.024); the **correct Poisson-Newton MAP is no better** (0.078) despite being calibrated ⇒
  **encoder fidelity ≠ estimator quality** (score identity). So: cheap MAP + optional quantile.
- **Negative result — quantile does NOT win in the MLE's own regime (simulation_1, 240 reps, `zqe_q`
  arm). RE-RUN 2026-06-22 with the paper setting WZS=0.5 + ridge l2=0.5/n** (λ=0.5; per-fit λ/n,
  consistency-preserving). Overall mean Procrustes: plain `zqe` 0.316 < `zqe_q` 0.340 < `gllvm`
  0.410; `zqe_q` beats `zqe` only 22% of reps. Still the same story: plain MAP `zqe` is the best
  W-encoder; quantile is **not** an efficiency lever (use it only for robustness/embeddings). NB the
  WZS=1.0/no-ridge numbers (zqe 0.218 / zqe_q 0.250 / gllvm 0.300) are superseded.
- Meta: consistency holds for *any* surrogate, so we are not locked in — present MAP as the
  recommended default and the quantile as a cheap refinement the framework *admits*.

## Simulation-1 (and -3) paper setting + figures (2026-06-22)

**Setting standardized to WZS=0.5, ridge l2=0.5/n (λ=0.5).** Loading scale 0.5 (was 1.0) keeps
Poisson rates / binomial probs unsaturated; ridge λ=0.5 added per-fit as **λ/n** (NOT a constant —
the 1/n is what keeps consistency; `L2_COEF/Y.shape[0]` in the sweeps). Applied to simulation_1
(Poisson) and simulation_3 (binomial "bernoulli" twin, BINOM_TRIALS=10 kept). Robustness sim_2 uses
no ridge. Paper §"Ridge stabilization" updated: λ=0.5 in loading-recovery studies (was 0.1, stale).

**Framing: "regime" not "turf"; "competitive in the MLE's own regime", never "we beat/destroy".**
The §exp-turf prose + intro bullet + summary edited accordingly (label id `sec:exp-turf` kept).

**New numbers (sim_1, 20 reps, mean Procrustes).** n=20: `zqe`≈0.50–0.58 vs `gllvm` 0.75–0.94
(gllvm catastrophe rate >0.5 is 75–100%, worst 1.5; zqe worst ≤1.03). n=100: tied (within 0.01–0.03;
gllvm a hair better at small p, zqe at large p). n=500: tied (gllvm marginal efficiency edge at p≥20,
but still one flier at p=10 = 0.52 vs zqe ≤0.25). The old WZS=1.0 catastrophe at n=500,p=100 (0.88)
is GONE with the smaller loadings — catastrophes now concentrate at n=20.

**Two paper figures generated** (scripts in `simulations/simulation_1/`):
- `make_paper_figure.py` → `paper/figures/sim1_procrustes_box.pdf` (Fig `fig:sim1box`): Procrustes
  boxplots, fliers shown — the reliability/tail story (gllvm fliers above 0.5 line at small n).
- `make_bias_figure.py` → `paper/figures/sim1_bias.pdf` (Fig `fig:ridge-bias`, in appendix ridge
  §): residual-vs-true-loading, pooled; ridge shrinkage slope −0.15 (n=20) → −0.04 (n=100) →
  −0.002 (n=500), vanishing as λ/n; gllvm unbiased. GB's requested bias diagnostic.
- DEFERRED idea (GB): scale ridge by √p too (more responses = more data); kept λ/n for now.

## Robustness / Huberization section plan (2026-06-21)

- **Huberize the Pearson residual** $r=(y-\mu)/\sqrt{V(\mu)}$ (clip $\psi_c$), per GLM family
  ($\sqrt{V}$ = the scale: Poisson $\sqrt\mu$, etc.). Standardize $\mu$/scale against the **model
  marginal via the fantasies we already draw** (or via the quantile-calibrated $\hat\mu$) — *not* a
  crude encoder $\hat\mu$.
- **Validity:** Huberization is *not* restricted to true scores (Huber/Hampel IF theory is general);
  "only scores" applies to *optimality* (the OBRE = Huberized-recentered score). The **fantasy
  centering supplies the Fisher-consistency correction $\mathbb E_\theta[\psi_c]$ for free** — the
  term Cantoni–Ronchetti (2001, JASA, robust GLM) must derive analytically and which is intractable
  in latent-variable models. With the **canonical $T$**, our centered equation *is* the score ⇒
  Huberizing it = the OBRE (optimal); general $T$ ⇒ B-robust + consistent.
- **Guardrail to state:** apply the *identical bounded map* (encoder **and** clip) to data and
  fantasies, no-grad — that symmetry is why the Huberized equation stays centered at $\theta_0$.
- Closest prior art / contrast: Cantoni & Ronchetti (2001); Hampel et al. (1986) for IF/OBRE.

## GP-PCA / fixed-ℓ scale-bank (Paper 2 / Discussion angle) (2026-06-21)

Frame the GP-GLLVM as a **count-native, scale-resolved "GP-PCA"** feature extractor. Distinctive
output = the **per-factor length-scale spectrum** (no PCA/SpatialPCA/MEFISTO gives it). Built-in
noise/signal split: factors with $\ell <$ spot pitch are sub-resolution nugget/noise; real programs
at $\ell \gtrsim$ pitch. **Fixed-ℓ scan:** instead of estimating $\ell$, fix a predefined ℓ grid →
a deterministic multi-scale filter bank (filters fixed, *what loads* at each scale learned). Fit the
bank **jointly** ($B{=}I$) so scales compete → complementary channels; quantile-calibrate each →
clean multi-channel embedding before a neural net. Cites/contrast: wavelets / scattering /
scale-space (those filter the raw signal; this extracts latent programs per scale on counts);
SpatialPCA, MEFISTO (Gaussian, $O(n^3)$). Composable (ZQE one-liners): group-lasso on $W$-columns
→ automatic factor/rank selection; L1 / Huber; confirmatory masks; mixed response families.

**Parked for Paper 2 — scalability in $q$ (deliberately NOT in the main paper; it muddies the
contribution).** With $B=I$, $\Sigma=\mathrm{blkdiag}(K_1,\dots,K_q)$, so: **generating**
($z=L_\Sigma\varepsilon$) is exactly per-factor (linear in $q$, machine-exact); the **encoder solve**
($A=\Sigma^{-1}+(W^\top W/s^2)\otimes I_K$) is factor-*coupled* for distinct $\ell_k$ → use
prior-preconditioned **CG** (sub-cubic; exact to 1e-12; crossover vs dense $(qK)^3$ at $q\approx16$,
2.5× at $q{=}32$ and widening; a block-Jacobi preconditioner lowers the crossover). *Shared*-kernel
case → exact Kronecker eigentrick (no CG). Verified prototype:
`playground/gp-gllvm/_blockscale_verify.py`. For the main paper, dense is fine at the $q\le12$ we use;
this only matters at large $q$ → Paper 2.

---

## Encoder-class scope: "smooth + globally identifying", not "any" (2026-06-27)

**The sellable scoping of the encoder-flexibility result.** Do NOT pitch "any encoder works"
(reads as a content-free / negative result a referee distrusts). The precise, positive claim:

> The ZQE decoder estimator is consistent for **any encoder that is smooth and remains an
> identifying z-map over the θ-region traversed** — not literally any encoder, and not only
> θ-tracking ones.

Two ingredients, cleanly separated:
- **Fisher-consistency is encoder-agnostic.** For *any* fixed map `e(·)`, θ* is a root of the
  estimating equation: at θ* the centering `E_θ[T(Y_q)·η(e(Y_q);θ)]` equals the data term (both
  under `f_{θ*}`), so they cancel exactly. The encoder need not track θ for θ* to be a zero.
- **Identification needs smooth + globally valid.** Smoothness → the Jacobian `A=E[ψ sᵀ]` is
  well-defined and the IFT/AN machinery applies. Global validity (the encoder stays an informative
  z-map across the region) → `A` stays nonsingular and the objective stays bounded.

**Why this sells (flips the paper's known weak spot):**
1. *It has content* — a checkable condition, not "anything goes."
2. *It predicts the experiments instead of apologizing for them.* Closed-form MAP satisfies the
   condition → works. The frozen amortized VAE is only *locally* valid (trained at θ_VAE) → violates
   global identification → diverges (`wz→±50`, Procrustes→1). The divergence becomes **confirmation
   of the boundary**, not an embarrassment. (Correction to the older "frozen ⇒ fails" note: frozen
   is NOT the disqualifier — *locally-trained* is; a smooth, globally-valid frozen encoder would
   also be consistent.)
3. *It's constructive + explains scale in one breath:* you only need the **cheapest** smooth global
   z-map (the parameter-free closed-form encoder) — no θ-tracking, no training — and dropping all
   encoder parameters is precisely what enables the large-scale regime.

Net: converts the defensive "encoder doesn't matter" into the offensive "here is the valid encoder
class, the cheap closed-form member qualifies, and discarding the rest is what buys scale." Sits
directly under the title *Z-Estimation for Large-Scale Latent Variable Models*.
