# Five-Minute Pitch — ZQE: Consistent Decoder Estimation Without Accurate Posterior Inference


---

## The Problem (1 min)

Latent variable models — GLLVMs, VAEs, factor models — are everywhere in genomics and ecology. The standard pipeline is: train an encoder to approximate the posterior, use it to update the decoder, iterate. The problem is that **a bad encoder biases the decoder, and that bias does not go away as n grows.** This is the amortization gap / variational bias, and it's well-documented. The deeper issue is structural: the standard ELBO update is only an unbiased estimating equation for the decoder when the posterior is exact. Use the wrong encoder and you're solving the wrong equation.

The field's response has been to make better encoders — normalizing flows, IWAE, semi-amortized methods. That's expensive, and it still couples encoder quality to decoder correctness. We ask a different question: **can you estimate the decoder consistently while treating the encoder as completely given and possibly wrong?**

---

## The Idea (1.5 min)

The MLE score in a latent variable model has a clean two-term structure (this is classical):

$$s(\theta; y) = \underbrace{\mathbb{E}_{p(z|y;\theta)}\bigl[D(z;\theta)^\top T(y)\bigr]}_{\text{data term}} - \underbrace{\mathbb{E}_{p(z|y;\theta)}\bigl[D(z;\theta)^\top \mu(z;\theta)\bigr]}_{\text{model term}}$$

The data term matches sufficient statistics; the model term is their expected value under the model. At the truth these balance exactly.

Now replace the true posterior $p(z|y;\theta)$ with any surrogate $q(z|y;\theta)$ — a pretrained encoder, a MAP approximation, anything. The data term becomes biased. The model term becomes biased. **But if you take the difference**, you get a new estimating equation whose population moment is zero at the truth $\theta^*$ — by the score identity — regardless of what $q$ is.

This is the **$Z_q$ estimator**: form the difference, not the individual terms. It's a centered Z-estimator. The surrogate controls efficiency and conditioning, not the population root.

Formally, the estimating equation is:

$$\frac{1}{n}\sum_i \mathbb{E}_{q(z|y_i;\theta)}\bigl[D^\top T(y_i)\bigr] = \frac{1}{n}\sum_i \mathbb{E}_{q(z|y_i;\theta)}\bigl[D^\top \mu(z;\theta)\bigr]$$

Match empirical sufficient statistics to their model-expected value under the *same* surrogate. The cross-centering kills the bias.

---

## The Canonical Instance: Gaussian Proxy (1 min)

The general theory works for any surrogate. The practically useful instance is the **Gaussian proxy**: choose $T(y)$ to be a variance-stabilising transform (e.g. Anscombe's $T(y) = 2\sqrt{y + 3/8}$ for Poisson data), so that $T(y) | z \approx \mathcal{N}(Wz + b, I)$. Then everything closes in analytical form:

- **Encoder**: $\hat{z}(y) = (W^\top W + \lambda I)^{-1} W^\top T(y)$ — one Cholesky solve, shared across observations. No neural network, no architecture choices.
- **Centering term**: $\mathbb{E}[T(Y)^\top \hat\eta(T(Y)) \mid z] = \mathrm{tr}(H) + \mu(z)^\top \hat\eta(\mu(z))$, where $H = W(W^\top W + \lambda I)^{-1}W^\top$ — closed form in $\theta$.
- **Objective**: fully deterministic and smooth in $\theta$ → **L-BFGS**, no learning rate, convergence in tens of steps.

The Gaussian proxy is the design principle; it decouples computational tractability from correctness. Even if $T(y)$ is not truly Gaussian, the estimating equation is still consistent.

---

## Theory (30 sec)

We prove consistency and asymptotic normality under a local Jacobian nonsingularity condition — the ZQE analogue of the information matrix condition in MLE theory. The Jacobian is a perturbation of the Fisher information, and we verify nonsingularity explicitly for Poisson GLLVMs. We also show that among all centered estimating equations of this form, ZQE minimises first-order sensitivity to posterior misspecification.

---

## Empirical Results (1 min)

Three controlled experiments:

1. **Misspecification sweep**: progressively reduce encoder quality in a Poisson GLLVM. Standard VA/Laplace (as in R's `gllvm`) drifts. ZQE stays flat. Measured by relative orthogonal Procrustes error on the loading matrix $W$.

2. **High-dimensional sweep** ($p \in \{10, 100, 1000, 2000\}$, $N=1000$, $q=10$): ZQE Procrustes error stays ~0.02–0.03 across the full range. The VAE collapses to 0.63 at $p=2000$. ZQE scales; the VAE doesn't.

3. **Transfer / frozen encoder**: train an encoder on a reference model, freeze it, then move the decoder. Standard ELBO updates exhibit persistent bias; ZQE recovers the perturbed decoder.

---

## Why It Matters (30 sec)

Once you have $\hat\theta$, you have a fully specified GLM. The encoder is $\hat{z} = (W^\top W + \sigma^2 I)^{-1} W^\top T(y)$ — closed form with analytic uncertainty. You can hand this to a biologist: confidence intervals on loadings, likelihood ratio tests on zero loadings, confirmatory structure (fixing known-zero entries is a one-line gradient mask). A VAE gives you a black-box posterior you cannot interrogate.

This is not a niche technical fix. It's a principled answer to a question that has been open since EM: **you do not need accurate posterior inference to estimate the decoder accurately**, provided the estimating equations are properly centered.

---

## Status & Target

- Theory complete (consistency, AN, Jacobian conditions, robustness optimality).
- Gaussian proxy + L-BFGS implemented and benchmarked.
- Target: JRSS-B, ~6 months.
- Happy to share the draft.
