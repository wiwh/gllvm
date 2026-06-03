# ZQE + Gaussian MAP for Negative Binomial — Design Notes

These notes collect ideas, options, and a minimal implementation plan for
adapting the ZQE + Gaussian MAP encoder + proximal-L1 workflow to a
Negative-Binomial (NB) response model.  The goal is to preserve the ZQE
cross-term objective and the proximal L1 trick for sparsity recovery while
handling NB heteroskedasticity.

---

## 1. High-level idea

- Replace the Poisson model with NB: mean `mu = exp(W z + b)` and dispersion
  `phi` (NB parameterisation).  Var(Y | z) = mu + mu^2 / phi.
- Keep ZQE cross-term: real term uses `h_dec(y)` and encoder-based `eta(zhat)`;
  centering term estimated via Monte Carlo sampling of `Y | z` (cheap under
  `no_grad`).
- Use a closed-form or approximate Gaussian MAP encoder so the encoder is
  cheap and differentiable w.r.t. `W` (but no gradient flows through the
  encoder outputs during the outer step).
- Apply proximal L1 to `W` after each SGD step (or ISTA-style step), same as
  for Poisson.

---

## 2. Encoder / h(y) options

1. Plug-in heteroskedastic Gaussian MAP (recommended first attempt)
   - Use a transform `h(y)` (e.g. `log1p`) but allow a diagonal noise covariance
     `Sigma(z)` that depends on `mu(z)`.
   - Weighted closed-form MAP:
     z_hat = (W^T Sigma^{-1} W + I)^{-1} W^T Sigma^{-1} (h(y) - b)
   - Compute `Sigma` by plug-in: use `mu(z) = exp(W z + b)` evaluated with
     the current `W` (inside `no_grad()` to make encoder a fixed map per
     outer-step).  The variance of `h(Y)` can be approximated via delta method
     or analytic expression when using simple `h`.
   - Pros: closed-form weighted linear solve, cheap (one QxQ solve per batch).
   - Cons: requires plugging in `mu(z)` and an approximation for Var[h(Y)|z].

2. Variance-stabilizing transform (VST)
   - Try NB-specific VSTs or robust transforms (`log1p`, `sqrt`) that reduce
     heteroskedasticity. If `h(y)` is close to homoskedastic, revert to the
     simple Gaussian MAP `(W^T W + lambda I)^{-1} W^T (h(y)-b)`.
   - Pros: simplest encoder; no per-sample Sigma.
   - Cons: approximation quality depends on `mu` and `phi`.

3. Iterative per-sample MAP (fallback)
   - If approximations fail, run an inner L-BFGS or Newton solve per observation
     to get the exact MAP in `h`-space (expensive but accurate).

---

## 3. Centering term

- Closed-form centering like Anscombe for Poisson is unlikely for general NB
  unless we craft a specific `h` with nice moments.  Therefore, keep the
  Monte-Carlo centering approach:

  - Draw many `Z_sim ~ N(0, I)` (or a batch per SGD step), then simulate
    `Y_sim ~ NB(mean = mu(Z_sim), phi)` and compute `h(Y_sim)` and `eta(zhat)`.
  - All of this is inside `torch.no_grad()` so you can massively increase the
    simulation budget (`N_SIM_MIN`) to reduce centering variance at little
    CPU/GPU cost.

- Optionally: if a VST with known mean/variance gives analytical moments,
  derive an approximate closed-form centering term. This is lower priority.

---

## 4. Dispersion (phi) handling

- Start with fixed `phi` (global) as a hyperparameter — simplest and stable.
- Later: make `phi` learnable (global or per-response). If learnable, treat
  it carefully: gradients through `phi` interact with `W` and `Sigma` used by
  the encoder; consider separate optimizer or telescoped inner-outer updates.

---

## 5. Proximal L1 and identification

- Proximal L1 (soft-threshold) on `W` is unchanged and remains the right
  technique for producing exact zeros and selecting a rotation.
- Identifiability arguments are identical: sparsity breaks rotational
  invariance and selects the rotation where `W` is sparse.

---

## 6. Practical minimal implementation plan

1. NB parameterisation: choose `mu = exp(W z + b)` and fix `phi` (e.g. 10).
2. Pick `h(y) = log1p(y)` as a robust starting transform.
3. Implement `GaussianMAPEncoder_NB` (plug-in Σ version):
   - For each sample in batch compute `hy = h(y)`.
   - Use current `W` to evaluate `mu(z)` for candidate `z` draws; compute
     Var[h(Y_j) | z] (delta method or approximate analytic) → diagonal
     `Sigma(z)`.
   - Compute rhs = (hy - b) @ W and solve: A = W^T Sigma^{-1} W + ridge * I,
     then z_hat = A^{-1} rhs^T (batched solves).
   - Return z_hat (no_grad) so gradient flows only through `W` in outer pass.
4. Keep existing ZQE loss; for centering, simulate many `Y_sim ~ NB(mu(Z_sim), phi)`
   (increase `N_SIM_MIN` if NS is small).
5. Reuse proximal L1 ISTA step after `opt.step()` unchanged.
6. Validate on synthetic NB data: simulate sparse W, run arms (no-L1, L1-loss,
   prox-L1), measure Procrustes and exact zeros.

---

## 7. Numerical pitfalls and notes

- Sigma diag entries can be very small or large (for extreme mu); clamp them to
  a stable range to avoid ill-conditioning of `A`.
- The plug-in `Sigma` approximation introduces bias; monitor encoder vs an
  inner-optimised ground-truth MAP on a small validation set to estimate bias.
- If `phi` is very small (huge overdispersion) you may need many more sims to
  stabilise the centering term and/or stronger L1 regularisation.
- A batched `torch.linalg.solve` or `cholesky` per batch is fine — cost is
  `O(Q^3)` per solve which is tiny for `Q` in the tens.

---

## 8. Testing strategy

- Implement a small synthetic NB simulator (same harness as current Poisson
  sims, but sampling from NB with specified `phi`).
- Run recovery experiments across `phi` ∈ {5, 10, 50}, `L1_LAMBDA` sweep,
  and `N_SIM_MIN` values (e.g., 2000, 8000, 20000) to find a stable regime.
- Compare `prox-L1` vs `L1-in-loss` vs no-L1 using Procrustes and exact-zero
  counts (same diagnostics as current notebook).

---

## 9. Recommendation

- Start with the plug-in heteroskedastic Gaussian MAP encoder + `log1p`.
- Keep `phi` fixed initially and use large `N_SIM_MIN` for centering.
- Use proximal L1 unchanged.
- Only fall back to per-sample inner L-BFGS MAP if those approximations fail.

---

If you want, I can implement the `GaussianMAPEncoder_NB` scaffold and the NB
simulation + experimental arms (identical diagnostics) and run a first
sweep.  Say the word and I'll add the code and notebook cells under
`dev/negativebinomial/`.
