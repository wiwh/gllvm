# ZQE with L-BFGS on Fixed Fantasy Samples

## Motivation

The current Adam+ReduceLROnPlateau training loop works, but the M-step is approximate: each epoch
uses freshly-drawn fantasy samples, making the objective stochastic. L-BFGS requires a deterministic
closure (it re-evaluates the loss multiple times per step for line search), so it cannot be applied
directly to the standard ZQE loop.

The fix is to decouple sample generation from optimisation: fix a batch of fantasy samples for
several L-BFGS steps, then refresh. This turns the inner loop into a clean EM algorithm where each
step is a genuine descent.

## Algorithm

```
Initialise W randomly (or from Gaussian warm start)

for outer = 1, ..., n_outer:

    ── E-step ────────────────────────────────────────────────────────────────
    Draw n_mc fantasy batches:  Y_q^(k) ~ p(Y | W),  k = 1..n_mc
    Solve Newton MAP for all (1 + n_mc) * N columns simultaneously:
        Z_obs  = argmax_z  log p(Y_obs | W, z)  - 0.5 ||z||^2
        Z_q^k  = argmax_z  log p(Y_q^k  | W, z) - 0.5 ||z||^2
    (Reuse the existing poisson_newton_batch vectorised solver.)
    Detach Z_obs, Z_q^k — they are fixed constants for the M-step.

    ── M-step ────────────────────────────────────────────────────────────────
    Run L-BFGS on W with closure:
        m1 = mean_n  sum_j log p(Y_obs[i,j] | W, Z_obs[i])
        m2 = mean_k mean_n  sum_j log p(Y_q^k[i,j] | W, Z_q^k[i])
        loss = -(m1 - m2)
        loss.backward(); return loss
    Iterate until L-BFGS line search converges (typically 20–50 closure calls).

    ── (optional) fantasy refresh ────────────────────────────────────────────
    Refresh fantasy samples every `refresh_every` outer steps (default: every step).
    Keeping samples fixed for multiple M-steps can save Newton solves at the cost
    of a slightly stale E-step.
```

## Why this is valid

- With Z fixed, `loss` is a deterministic function of W → L-BFGS line search is safe.
- The outer loop is an EM-style coordinate descent on the ZQE objective:
  E-step maximises over Z (MAP), M-step maximises over W.
- Each outer iteration is a true descent (up to Monte Carlo noise from fantasy samples).
- No learning rate to tune: L-BFGS has its own Wolfe-condition line search.

## Expected advantages over Adam

| | Adam + ReduceLROnPlateau | L-BFGS fixed-seed |
|---|---|---|
| M-step quality | approximate (1 gradient step) | exact (to L-BFGS tolerance) |
| LR tuning | patience / factor heuristic | none (line search) |
| Outer iterations needed | ~1000–1500 | ~50–200 |
| Wall time per outer iter | fast | ~10–50× slower (multiple closures) |
| Total wall time | ? | likely faster for small N, p |

## Key implementation notes

- Use `torch.optim.LBFGS(params, lr=1.0, max_iter=20, line_search_fn="strong_wolfe")`.
- The closure must call `opt.zero_grad()`, recompute loss from the **fixed** Z tensors, call
  `loss.backward()`, and `return loss`. Do not re-draw fantasy samples inside the closure.
- Z tensors must be `.detach()`ed before the M-step; they must not appear in the autograd graph.
- Gradient clipping is not needed (L-BFGS uses its own damping), but consider adding a
  `max_norm` check as a safety guard for the first few outer steps.
- `refresh_every=1` (default): new fantasy batch every outer step — highest variance, most
  exploration.  `refresh_every=k` for k>1: amortise Newton cost over k M-steps.

## Suggested hyperparameters for first experiment

```python
n_outer      = 200        # outer EM iterations
n_mc         = 10         # fantasy samples per outer step (more than Adam: cheaper per step)
lbfgs_iters  = 20         # max L-BFGS iterations per outer step
refresh_every = 1         # refresh fantasy samples every outer step
```

## Comparison plan

Run in the same notebook framework as `zqe_poisson_map.ipynb`:
- Same simulation: NL=5, NR=100, NS=100, ACT=NL, SEED=42
- Same decoder: PoissonLog1pGLM
- Same encoder for E-step: PoissonMAPEncoderNewton (exact MAP)
- Plot: Procrustes vs outer iteration (not epoch), wall time on x-axis optional
- Compare final Procrustes against Adam+ReduceLROnPlateau and R gllvm VA
