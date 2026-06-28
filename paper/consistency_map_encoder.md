# Consistency of the $Z_q$ estimator with the (true-model) MAP encoder

*Proof note — math first, prose/LaTeX later. Aligns with `main.tex` (eqs.
`psi_theta_dep`, `jacobian_final`; Assumptions `uniform-conv`, `nonsingularity`;
Thm `Zq-main`) and the Gaussian-proxy identification in `paper/CLAUDE.md`
(Lemma 1 b-block, Lemma 2 W-block large-$p$).*

---

## 0. What is new vs. the Gaussian-proxy appendix

For the Gaussian proxy the encoder is the **closed-form** ridge map
$\hat z_G(y;\theta)=(W^\top W+\sigma^2 I)^{-1}W^\top(T(y)-b)$, so smoothness in
$\theta$ and the Jacobian are immediate. Here the surrogate is the **true-model
penalized MLE of $z$** (the Poisson–Newton MAP),

$$
\hat z(y;\theta)\;=\;\arg\max_{z\in\mathbb R^q}\Big\{\log f_{Y\mid Z}(y\mid z;\theta)-\tfrac{\lambda}{2}\|z\|^2\Big\},
$$

defined **implicitly** as an $\arg\max$. Two — and only two — things must be supplied
beyond the general theory:

- **(A) Regularity of the implicit encoder** (continuity / $C^1$ in $\theta$, and an
  $L^2$ envelope) so that Assumption `uniform-conv` and the differentiability part of
  Assumption `nonsingularity` hold. *This is the IFT step.*
- **(B) Local nonsingularity** of the Jacobian $A(\theta_0)$ for the MAP-induced
  $\psi$. *Here the "MAP is consistent for $z$ as $p\to\infty$" fact is used — and it
  is cleaner than the proxy because the true MAP carries no linearisation bias.*

Everything else is inherited verbatim from the general framework. In particular
**consistency in $n$ requires no encoder derivatives**: by the Jacobian-cancellation
identity (`main.tex` eq. `jacobian_final`) the $\partial\hat z/\partial\theta$ terms
drop out of $A(\theta_0)$. The IFT is needed for *regularity and computation*, not to
make the population root correct.

---

## 1. Setup and the MAP surrogate

Conditionally-independent exponential-family GLLVM (notation as in `CLAUDE.md`
§Identification): for $j=1,\dots,p$,

$$
f_j(y_j\mid z;\theta)=\exp\{\tilde T_j(y_j)\,\eta_j-A_j(\eta_j)+B_j(y_j)\},\qquad
\eta_j=\eta_j(z;\theta)=w_j^\top z+b_j,
$$

prior $Z\sim\mathcal N(0,\lambda^{-1}I_q)$ (so the MAP penalty is $\tfrac\lambda2\|z\|^2$;
$\lambda=1$ for the canonical $\mathcal N(0,I)$), $\theta=(W,b)$. Write
$\mu_j(\eta)=A_j'(\eta)=\mathbb E[\tilde T_j(Y_j)\mid \eta]$ and
$v_j(\eta)=A_j''(\eta)=\operatorname{Var}(\tilde T_j(Y_j)\mid\eta)>0$.

The $Z_q$ statistic is a (possibly different) $T(y)=(T_1(y_1),\dots,T_p(y_p))^\top$,
e.g. $T_j=\log(1+\cdot)$ for Poisson — **decoupled** from the canonical $\tilde T_j$
that defines the likelihood/MAP. The MAP surrogate is the point mass
$q_{Z\mid Y;\theta}(\cdot\mid y)=\delta_{\hat z(y;\theta)}$, giving (`main.tex`
eq. `psi_theta_dep` with $D=\partial\eta/\partial\theta$)

$$
\psi(\theta;y)=D\big(\hat z(y;\theta);\theta\big)^\top T(y),\qquad
D(z;\theta)=\partial\eta/\partial\theta .
$$

**MAP first-order condition.** $\hat z=\hat z(y;\theta)$ solves
$$
G(z,\theta,y):=\nabla_z\Big[\log f_{Y\mid Z}-\tfrac\lambda2\|z\|^2\Big]
= W^\top\big(\tilde T(y)-\mu(Wz+b)\big)-\lambda z = 0,
\tag{FOC}
$$
where $\tilde T(y)=(\tilde T_j(y_j))_j$ and $\mu(\cdot)$ acts coordinatewise.

---

## 2. Inherited (encoder-agnostic) facts

These hold for **any** surrogate, the MAP/$\delta$ included; no proof needed here, we
just record what we are standing on.

- **(Centering / Fisher consistency.)** $\Psi(\theta_0)=\mathbb
  E_{\theta_0}[\psi(\theta_0;Y)]-\mathbb E_{\theta_0}[\psi(\theta_0;Y)]=0$
  (`main.tex` eq. after `psi_theta_dep`). $\theta_0$ is a population root regardless of
  encoder quality.
- **(Jacobian cancellation.)** $A(\theta_0):=-\nabla_\theta\Psi(\theta_0)=\mathbb
  E_{\theta_0}[\psi(\theta_0;Y)\,s(\theta_0;Y)^\top]$ (`main.tex` eq.
  `jacobian_final`), where $s$ is the true marginal score. **All
  $\partial\hat z/\partial\theta$ terms cancel** — the proof is free of encoder
  derivatives.

So the MAP-specific work is exactly (A) regularity and (B) nonsingularity of this
$A(\theta_0)$.

---

## 3. Lemma A — the MAP encoder is well-posed and $C^1$ (IFT)

**Claim.** Fix a neighbourhood $\Theta_0\ni\theta_0$ on which $W$ has full column rank.
For every $y$ in the support and every $\theta\in\Theta_0$:

1. the penalized log-posterior $z\mapsto \log f_{Y\mid Z}(y\mid z;\theta)-\tfrac\lambda2\|z\|^2$
   is **strictly concave**, so $\hat z(y;\theta)$ exists and is **unique**;
2. $\hat z(\cdot)$ is **continuously differentiable** in $(\theta,y)$ on $\Theta_0$, with
   $$
   \frac{\partial \hat z}{\partial\theta}
   = -\Big(\underbrace{W^\top V(\hat z)\,W+\lambda I_q}_{=\,-\,\partial G/\partial z\;\succ 0}\Big)^{-1}\frac{\partial G}{\partial\theta},
   \qquad V(z)=\operatorname{diag}\big(v_j(\eta_j(z;\theta))\big);
   $$
3. consequently $\psi(\theta;y)=D(\hat z(y;\theta);\theta)^\top T(y)$ is continuous in
   $\theta$, and $\theta\mapsto A(\theta)=\mathbb E_{\theta_0}[\psi(\theta;Y)s(\theta_0;Y)^\top]$
   is continuous (hence Assumption `nonsingularity`'s $C^1$ requirement and Assumption
   `uniform-conv` hold under the envelope below).

**Proof.**

*(1) Strict concavity.* The Hessian of the penalized objective in $z$ is
$$
\partial_z G(z,\theta,y) = -\,W^\top \operatorname{diag}\!\big(A_j''(\eta_j)\big) W-\lambda I_q
= -\big(W^\top V(z)W+\lambda I_q\big).
$$
Each $A_j''=v_j>0$ (non-degenerate exponential family), so $W^\top V W\succeq 0$ and
$W^\top V W+\lambda I_q\succeq\lambda I_q\succ0$ for any $\lambda>0$. Thus the objective is
strictly concave with a globally bounded-above, coercive shape (the $-\tfrac\lambda2\|z\|^2$
term dominates at infinity), so a unique maximizer $\hat z$ exists. *Note:* the ridge
$\lambda>0$ is what guarantees invertibility even when $W^\top V W$ is rank-deficient or
the data separate — i.e. exactly the configurations where the unpenalized $z$-MLE would
fail to exist. This is the role of the prior.

*(2) IFT.* $G$ is $C^1$ in $(z,\theta,y)$ (compositions of $A_j',w_j^\top z+b_j$, smooth on
the exponential-family support). By (1), $\partial_z G=-(W^\top V(\hat z)W+\lambda I_q)$ is
nonsingular at $(\hat z,\theta,y)$. The implicit function theorem gives a unique $C^1$ map
$\hat z(\theta,y)$ with the stated derivative. (For $\partial G/\partial\theta$: differentiate
$G$ in $W$ and $b$ holding $z$; e.g. $\partial G/\partial b=-W^\top V(\hat z)$.)

*(3) Continuity + envelope.* $\psi$ is a composition of the $C^1$ map $\hat z$ with the
smooth $D,\eta$ and the fixed $T(y)$, hence continuous in $\theta$. The MAP is bounded by
$\|\hat z\|\le \lambda^{-1}\|W^\top(\tilde T(y)-\mu(b))\|$ at the stationary point (from
(FOC), using monotonicity of $\mu$), so $\|\psi(\theta;Y)\|\le g(Y)$ for an envelope
$g$ with $\mathbb E_{\theta_0}[g(Y)^2]<\infty$ under the family's second-moment condition
on $T,\tilde T$. Dominated convergence then gives continuity of $A(\cdot)$ and, with a
Glivenko–Cantelli class for $\{\psi(\theta;\cdot):\theta\in\Theta_0\}$ (compact $\Theta_0$
+ continuity + envelope), the ULLN of Assumption `uniform-conv`. $\square$

---

## 4. Lemma B — centering is exact for the $\delta$ (MAP) surrogate

Immediate special case of the general centering identity: with
$q_{Z\mid Y;\theta}=\delta_{\hat z(y;\theta)}$ the same deterministic map $\hat z(\cdot;\theta)$
is applied to data ($Y\sim f_{\theta_0}$) and to fantasies ($Y\sim f_{\theta}$), so
$\Psi(\theta_0)=0$. Recorded for completeness; nothing MAP-specific. $\square$

---

## 5. Proposition — local nonsingularity of $A(\theta_0)$ for the MAP encoder

We give two routes; either suffices for Assumption `nonsingularity`.

### 5a. General perturbation condition (any $p$, numerically checkable)

Writing $\mathcal I(\theta_0)=\mathbb E_{\theta_0}[s\,s^\top]$ for the marginal Fisher
information, $A(\theta_0)=\mathcal I(\theta_0)+R$ with
$R=\mathbb E_{\theta_0}[(\psi-s)\,s^\top]$. Hence $A(\theta_0)$ is nonsingular whenever
$$
\lambda_{\min}\!\big(\mathcal I(\theta_0)\big)\;>\;\|A(\theta_0)-\mathcal I(\theta_0)\|_{\mathrm{op}},
$$
which is verifiable at any given $\theta_0$ by Monte Carlo (draw $Y\sim f_{\theta_0}$,
form $\psi$ via the MAP solve and $s$ via the conditional-score formula). Same structure
as the Gaussian-proxy Proposition part (3), but with the MAP $\psi$.

### 5b. Large-$p$ route (the clean one — true MAP beats the proxy)

The key sub-result is that the **true-model MAP recovers $z$**:

**Step 1 (MAP $z$-consistency).** At the truth $Z$, the per-response score has conditional
mean zero: $\mathbb E[\tilde T_j(Y_j)-\mu_j(w_j^\top Z+b_j)\mid Z]=0$. So $Z$ is the
population root of the unpenalized $z$-estimating equation
$\tfrac1p\sum_j(\tilde T_j(y_j)-\mu_j(\eta_j))w_j=0$. Under
**(A1)** $\lambda_{\min}(W^\top V W)\ge c\,p$ and **(A2)** $\max_j\|w_j\|^2/p\to0$
(the spectrum / leverage conditions from the W-block lemma), $\hat z$ is the
$M$-estimator of a fixed-dimensional $z\in\mathbb R^q$ from $p$ conditionally-independent
responses, so
$$
\hat z(Y;\theta_0)=Z+\underbrace{O_{L^2}(p^{-1/2})}_{\text{noise around the root}}+\underbrace{O(\lambda/p)}_{\text{ridge shrinkage}} .
$$

> **Why cleaner than the proxy.** The Gaussian-proxy encoder carried an extra
> *linearisation-bias* term $\beta_j(Z)=\mathbb E[T_j(Y_j)\mid Z]-(w_j^\top Z+b_j)$
> (its assumption (A4), $O(p^{-1/2})$ in the W-block proof). The true MAP uses the
> *correct* $\mu_j$, so $\beta_j\equiv0$: **no linearisation bias**. The only departure
> from $Z$ is the $O(\lambda/p)$ ridge shrinkage and the irreducible $O(p^{-1/2})$ noise.
> This is the asymptotic shadow of the empirical finding that the Poisson-MAP encoder
> overtakes the Gaussian-log1p MAP at large $p$.

**Step 2 (Jacobian → Fisher).** $\psi(\theta_0;Y)$ and $s(\theta_0;Y)$ are both linear in
their latent argument through the exponential-family structure, so
$\|\psi-s\|\le L\|\hat z-Z\|$ for an $L=L(\theta_0)$. With $\delta_p:=\hat z-Z=O_{L^2}(p^{-1/2})$,
the per-feature normalisation $\bar\psi=p^{-1/2}\psi,\ \bar s=p^{-1/2}s$ gives
$\bar A(\theta_0)=\mathcal I(\theta_0)/p+\bar R_p$ with $\|\bar R_p\|_{\mathrm{op}}=O(p^{-1/2})$
(Cauchy–Schwarz, as in the W-block Step 2). Since $\mathcal I(\theta_0)/p$ has eigenvalues
bounded below by $c'>0$ under (A1), $\bar A(\theta_0)$ — hence $A(\theta_0)=p\,\bar A$ — is
nonsingular for all $p$ large enough. $\square$

**Caveat (global vs. local).** Nonsingularity is *local*. On heavy-tailed draws the
Poisson-MAP $z$-equation can have a spurious root (collapse), a *global* identification
failure of the surrogate, not a violation of Lemma A or 5b. Mitigants: the encoder ridge
$\lambda>0$ (Lemma A: keeps the MAP well-posed) and the $O(c/n)$ Tikhonov ridge on $W$
(below). Generate sim data on CPU for reproducibility (see project notes).

---

## 6. Theorem — consistency and asymptotic normality (MAP encoder)

**Statement.** Consider the exponential-family GLLVM of §1 with the true-model MAP
surrogate, encoder ridge $\lambda>0$, and $T_j$ admitting finite second moments. Assume
the spectrum/leverage conditions (A1)–(A2) (for the large-$p$ route) *or* the verifiable
perturbation condition of §5a. Then Assumptions `uniform-conv` and `nonsingularity` hold,
and by `main.tex` Thm `Zq-main` there is a sequence of roots $\hat\theta_n$ of $\Psi_n=0$ with
$$
\hat\theta_n\xrightarrow{p}\theta_0,\qquad
\sqrt n(\hat\theta_n-\theta_0)\Rightarrow\mathcal N(0,\Sigma),\quad
\Sigma=A(\theta_0)^{-1}B(\theta_0)A(\theta_0)^{-\top},
$$
with $A(\theta_0)=\mathbb E_{\theta_0}[\psi s^\top]$ (eq. `jacobian_final`) and
$B(\theta_0)=\operatorname{Var}_{\theta_0}(\psi(\theta_0;Y))$ (centered-equation form).

**Proof.** Lemma A ⇒ Assumption `uniform-conv` and the $C^1$/continuity half of Assumption
`nonsingularity`. Lemma B ⇒ $\theta_0$ is the population root. Proposition §5 ⇒ the
nonsingularity half. Apply Thm `Zq-main`. $\square$

---

## 7. Remarks

1. **IFT is reused as the optimisation tool.** The same $\partial\hat z/\partial\theta$
   from Lemma A(2) lets L-BFGS / Newton-CG differentiate through the *implicit* MAP solve
   without unrolling the Newton iterations — the non-Gaussian analogue of the
   closed-form proxy Jacobian (cf. the existing "IFT for encoder Jacobians" remark).
   But, per §2, it is **absent from the consistency proof**.

2. **Two ridges, two jobs.** (i) Encoder ridge $\lambda>0$ on $z$: guarantees strict
   concavity ⇒ a unique, well-posed, $C^1$ MAP (Lemma A) — a *regularity/identifiability*
   device, $O(\lambda/p)$ shrinkage in $\hat z$. (ii) Loading ridge $c/n$ on $W$
   (Tikhonov stabiliser): a *finite-sample* fix for near-unidentified $W$ whose bias is
   $O(1/n)\ll O(n^{-1/2})$ SE, hence asymptotically negligible — does not enter the limit.
   They are independent.

3. **Encoder fidelity $\ne$ estimator quality.** Consistency holds for *any* surrogate
   (Lemma B); the MAP's $z$-recovery (§5b Step 1) is used only to *certify large-$p$
   nonsingularity* and to explain efficiency, not to make $\hat\theta_n$ consistent. This
   is the honest statement of the user's "MAP=MLE+ridge ⇒ consistent" intuition: the MAP
   being good buys *identification/efficiency*, while *consistency is bought by centering*.

4. **Relation to the Gaussian-proxy result.** Replace the closed-form $\hat z_G$ by the
   implicit MAP: §3 supplies the smoothness that was free before; §5b drops the
   linearisation-bias term $\beta_j$, giving a strictly cleaner large-$p$ Jacobian. The
   b-block diagonal-PD argument (`CLAUDE.md` Lemma 1) carries over with $\hat z_G\to\hat z$
   wherever the encoder enters $\psi$; re-derive its covariance with the MAP $\psi$ when
   writing the appendix (TODO: explicit b-block for the MAP).

---

## 8. TODO before LaTeX

- Verify the b-block (diagonal, PD) explicitly for the MAP $\psi$ (the §CLAUDE Lemma 1
  analogue) — likely identical structure since it rests on conditional independence +
  score orthogonality, not on the encoder's closed form.
- State the precise moment/identifiability hypotheses on $(T,\tilde T)$ as a numbered
  assumption block matching `main.tex` style.
- Decide placement: a new appendix subsection "Identification under the true-model MAP
  encoder", parallel to the Gaussian-proxy one.
