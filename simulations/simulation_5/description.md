# Simulation 5 — GP-GLLVM at scale via K-subsets (composite ZQE)

**Main-paper headline: a Gaussian-process latent GLLVM with large, fully-correlated
groups, fit in seconds — because fitting only ever touches a small random K-subset
of each group (GP marginal theorem), never the big within-group covariance.**

## Model

`B` groups, each with `G` observations on a grid `t` (timepoints / locations).
Within a group **all `G` observations are GP-correlated** through the latent; groups
are independent. Poisson (log-link):

```
per group:  ε ~ N(0, I_{G×q}),  z = L(ℓ)ε  (L = chol K(t), G×G),  y ~ Poisson(e^{Wz+b}).
```

The data is generated **once** with full within-group correlation (one `G×G`
Cholesky, feasible for `G` up to thousands).

## The trick: fit on K-subsets (K ≪ G), tune K

By the **GP marginal theorem**, any subset of the `G` points is itself an exact GP
draw `N(0, K(t_subset))`. So during fitting we sample, per step, a random **K-subset**
of each group's points and a minibatch of groups — every operation is a `K×K` block.
**`K` is a hyperparameter** trading cost for efficiency. The kernel lives in the
**decoder** (`ε → z = L(ℓ)ε → η`), so `ℓ` is estimated by plain backprop; the encoder
is no-grad imputation (per-obs MAP → whiten `ε̂=L(ℓ)⁻¹Ẑ` on the K-subset). Plain
centered ZQE `−(m1−m2)`; fantasies are also K-subsets. **Per-step cost
`O(batch·K·p + K³)` — independent of both `G` and `B`.**

(Block subsetting for GPs is the Vecchia / composite-likelihood idea; the
contribution is its *likelihood-free* realization for **non-Gaussian** GLLVMs — the
block likelihood is intractable, but sample+impute is trivial.)

## Why others can't

A full GP latent over a group costs `O(G³)`; over all `n=B·G` it is hopeless. The
marginal/block likelihood integrates a non-Gaussian latent — no closed form. And
`gllvm` has **no GP latent at all** (the kernel `ℓ` is not even a parameter it has).

## Validated results (q=2, p=30, RBF kernel, true ℓ=2.0; GPU; B=2000, fit K=15)

| group size G | n = B·G | est ℓ | loadings Procrustes | **fit time** |
|---|---|---|---|---|
| 50 | 100 000 | 1.87 | 0.047 | 3.6 s |
| 200 | 400 000 | 1.96 | 0.015 | 3.5 s |
| 1 000 | **2 000 000** | 2.03 | 0.019 | **3.6 s** |

Recovery *improves* with `G` (richer correlation); **fit time is flat in `G`** — the
marginal-theorem payoff. The `K` knob (at G=200): K=5 → 0.030/2.6 s; K=15 →
0.015/3.1 s; K=40 → 0.023/4.0 s. Lengthscale recovery holds across true ℓ.

## What sim 5 reports

`scaling.ipynb`: (i) recovery + **flat fit time vs group size `G`** (the headline);
(ii) the `K`-subset cost/accuracy tradeoff; (iii) scaling in the number of groups
`B`; (iv) lengthscale recovery across true ℓ. Implementation: `gp.py` (self-contained
model + `fit_gp_zqe` with `subset_K`). `q` is a free knob (2 here; larger later).
