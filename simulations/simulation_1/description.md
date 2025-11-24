# Experiment: Robustness of ZQE to Encoder Misspecification  
**Demonstrating that ZQE remains consistent even when the encoder is severely misspecified, while VI collapses.**

This experiment creates a **2D grid** over:

1. **Encoder misspecification severity** (shrink encoder width)
2. **Sample size** (small to large datasets)

For every configuration in the grid, we fit:

- **Variational Inference (VI / VAE baseline)**
- **ZQE with ELBO-trained encoder**
- **ZQE with synthetic-data–trained encoder (optional)**

We then compare parameter recovery, estimating-equation residuals, identifiability, and likelihood.

---

# 1. Model Used: GLLVM

We use a Generalized Latent Variable Model with a mixture of GLM observation families, because:

- the likelihood is known,
- the latent posterior is nontrivial (non-Gaussian, non-conjugate),
- parameter recovery is cleanly evaluable,
- misspecification is easy to induce in the encoder,
- the ground truth posterior allows clean benchmarking.

---

# 2. Experiment Grid

We vary **encoder width** \( h \) and **sample size** \( n \):

### Encoder widths:
\[
h \in \{128, 64, 32, 16, 8, 4, 2, 1\}
\]

Small values produce catastrophic amortization bias.

### Sample sizes:
\[
n \in \{200, 500, 1000, 2000, 5000, 10000\}
\]

Larger datasets reveal asymptotic behavior.

---

# 3. Methods Compared

For each pair \((h, n)\), we fit:

### **(1) Variational Inference (VI / VAE-like)**  
- Encoder is an MLP of width \( h \)
- Decoder is GLLVM
- Parameters learned by ELBO
- Known to suffer amortization bias under encoder misspecification

### **(2) ZQE (with ELBO-trained encoder)**  
- Encoder first trained by ELBO, *exactly like VI*  
- Decoder then updated by ZQE:
  - No gradients through encoder  
  - Only uses samples from encoder  
  - Bias removed by centered Z estimation

### **(3) ZQE (encoder trained by synthetic data)** *(optional)*  
- Generate arbitrarily many synthetic \((Z, Y)\) pairs from decoder  
- Train encoder to regress \(Y \mapsto Z\)  
- This decouples encoder/decoder entirely  
- Illustrates extreme robustness of ZQE

---

# 4. Metrics Recorded

For each fitted model:

### **1. Parameter estimation error**
\[
\|\hat\theta - \theta_0\|_2
\]

Assesses correctness of decoder parameters.

---

### **2. Estimating-equation residual**
\[
\|\bar\psi\|_2
\]

- VI: does **not** go to zero with \( n \) if encoder misspecified  
- ZQE: always decreases with \(n\)

This directly visualizes the **core theoretical claim**.

---

### **3. Smallest singular value of the Jacobian**
\[
\sigma_{\min}\big(A_q(\hat\theta)\big)
\]

Measures **identifiability** of the estimating equations.

Expected behavior:

| Encoder size ↓ | VI Jacobian | ZQE Jacobian |
|----------------|------------|---------------|
| well-specified | non-singular | non-singular |
| medium | deteriorates | stable |
| tiny | collapses | remains identifiable |

This matches the nonsingularity requirement from theory.

---

### **4. Held-out NLL (IWAE estimate)**
We estimate:
\[
-\log p(Y \mid \hat\theta) \approx \text{IWAE}(k=500)
\]

ZQE usually improves or matches VI.

---

# 5. Visual Outputs

For each method:

### **Heatmap 1 – parameter error**
- x-axis: encoder width  
- y-axis: sample size  
- cell: RMSE of \(\hat\theta\)

### **Heatmap 2 – estimating-equation residual**
- Shows VI → biased plateau  
- ZQE → decreases with \( n \)

### **Heatmap 3 – Jacobian σᵐᵢₙ**
- VI → degeneracy under small encoder  
- ZQE → stays identifiable

### **Heatmap 4 – NLL (IWAE-500)**  
Optional but strengthens results.

---

# 6. Why This Experiment Is a Perfect Demonstration

This setup isolates exactly the theoretical contribution:

- VI performs well **only** when encoder is rich enough  
- VI collapses when encoder is bottlenecked  
- ZQE remains:
  - **consistent**  
  - **unbiased**  
  - **well-identified**  
  - **robust to posterior misspecification**

All within the same generative modeling framework.

GLLVM ensures clarity, interpretability, and complete control.

---

# 7. Optional Additions

- Track encoder posterior error:
  \[
  \text{KL}(q(z|y) \| f(z|y))
  \]
- Scatterplots comparing recovered parameters to truth  
- Surface plots of ZQE convergence trajectories  
