# CLAUDE.md — simulations/simulation_sparse

## Purpose

Experiment **A**: sparse loadings GLLVM.
Compares VAE vs ZQE estimators on recovery of a loading matrix W with
`active_latent=ACT` non-zero columns per latent dimension.

Main notebook: `exp_A_sparse_loadings.ipynb`

---

## Current experimental setup (Cell 2)

```python
NL=5, ACT=3, NR=2000, NS=100, WZS=0.3
```
- 5 latent dims, 3 active per dim, 2000 Poisson responses, 100 samples (train=85, val=15)
- Loading scale 0.3 (weak signal)
- Large-p / small-n regime — the key regime for the paper

### Evaluation metric
**Procrustes error** (`scipy.spatial.procrustes`). Lower = better.

---

## Methods compared (7 arms + VAE)

| Arm | Encoder type | T(y) | Learnable params |
|-----|-------------|------|-----------------|
| VAE | `Log1pEncoder` MLP (hidden=176) | log1p input | ~350k |
| MAP log1p | `MapEncoderGaussianLog1p` | log1p(y) | 0 |
| MAP sqrt | `MapEncoderGaussianLog1p` | sqrt(y) | 0 |
| MAP log1p+sqrt | `MapEncoderGaussianLog1p` | (log1p+sqrt)/2 | 0 |
| Gauss log1p | `GaussianPosteriorEncoderLog1p` | log1p(y) | 0 |
| Gauss sqrt | `GaussianPosteriorEncoderLog1p` | sqrt(y) | 0 |
| Gauss log1p+sqrt | `GaussianPosteriorEncoderLog1p` | (log1p+sqrt)/2 | 0 |
| Tracking log1p+sqrt | `TrackingEncoderLog1p` (MLP retrained on synthetic each epoch) | (log1p+sqrt)/2 | ~350k |

### Encoder types
- **MapEncoderGaussianLog1p**: closed-form MAP z = Σ_z W^T T(y), 0 params, O(p·q)
- **GaussianPosteriorEncoderLog1p**: same but full Gaussian posterior, 0 params, O(p·q)
- **Log1pEncoder**: MLP with log1p input normalisation (prevents NaN at large p)
- **TrackingEncoderLog1p**: MLP retrained each epoch on synthetic (y_q, z_q) from current θ (MSE)

---

## Known results (NL=5, ACT=3, seed=42)

### NR=2000, NS=100 (large-p regime)

| Method | Procrustes |
|--------|-----------|
| MAP log1p(y) | **0.286** |
| Tracking log1p+sqrt | 0.298 |
| Gauss sqrt(y) | 0.300 |
| Gauss log1p+sqrt | 0.306 |
| MAP log1p+sqrt | 0.314 |
| MAP sqrt(y) | 0.320 |
| Gauss log1p(y) | 0.323 |
| **VAE** | **0.740** ← 2.6× worse |

### NR=200, NS=100

| Method | Procrustes |
|--------|-----------|
| Gauss log1p+sqrt | **0.502** |
| VAE | 0.556 |

### NR=20, NS=100 (small-p regime)

| Method | Procrustes |
|--------|-----------|
| MAP log1p(y) | 0.533 |
| Gauss log1p+sqrt | 0.502 |
| VAE | 0.556 |

---

## Key scientific finding: large-p / small-n concentration

**ZQE improves dramatically as p grows; VAE does not.**

The Gaussian proxy encoder computes z ∝ W^T T(y), a sum over p features.
By LLN-in-p this concentrates toward the true posterior mean at rate 1/√p,
**for free, regardless of n**. At NR=2000 the proxy is already an excellent
posterior approximation. The VAE encoder has ~350k params to learn from 85
observations — severely data-starved, ELBO landscape degenerate.

The Tracking encoder (same 350k params, retrained on synthetic data each epoch)
is **no better** than the zero-parameter Gaussian proxy at NR=2000.
Concentration wins over expressivity.

Paper section already drafted: main.tex §"The large-p, small-n regime".

---

## VAE training protocol (robust at large p)

1. `Log1pEncoder`: applies log1p(y) before MLP — prevents NaN from raw counts
2. `VAE_HIDDEN = max(64, min(256, int(NR**0.5) * 4))` — scales hidden with √NR
3. Warmup: `max(500, NR//4)` epochs at lr=1e-2
4. 4 fine-tune stages: (3e-3/100, 1e-3/120, 3e-4/150, 1e-4/200) with early stopping
5. Early stopping threshold: 0.05 nats on val ELBO, budget 3000 ep/stage

---

## ZQE training protocol

- Adam on g.parameters() only (encoder in torch.no_grad())
- ReduceLROnPlateau(factor=0.5, patience=60, min_lr=1e-5)
- lr_init=5e-2, grad clip max_norm=5.0, 1000 epochs

---

## GLM classes (src/gllvm/glms.py)

| Class | T(y) |
|-------|------|
| `PoissonLog1pGLM` | log1p(y) |
| `PoissonSqrtGLM` | sqrt(y) |
| `PoissonLog1pSqrtGLM` | (log1p(y) + sqrt(y)) / 2 |
| `PoissonMixedTGLM` | y + log1p(y) |
| `PoissonMultiTGLM` | (log1p + sqrt + y/(1+y)) / 3 |

---

## Scalability (scRNA-seq: p~20k, n~1k)

| Encoder | Params | θ-consistent? | Scales? |
|---------|--------|--------------|---------|
| Gaussian proxy | **0** | ✓ always | ✓ O(p·q) |
| TrackingEncoder | ~2.5M (20k×128) | ✓ approx | ✗ too large |
| VAE encoder | ~2.5M | ✗ amortisation gap | ✗ data-starved |

---

## Next steps

- [ ] **p-sweep figure**: NR ∈ {20, 100, 500, 2000}, fixed NS=100 — ZQE improves monotonically, VAE flat/worse → key paper figure
- [ ] **Multi-seed** (seeds 0–4): get mean ± std, confirm not seed-specific
- [ ] Consider real scRNA-seq run (pbmc4k in notebooks/data/) as qualitative validation
