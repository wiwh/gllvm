import nbformat as nbf
nb = nbf.v4.new_notebook()
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def co(s): C.append(nbf.v4.new_code_cell(s))

md(r"""# GP-GLLVM on real spatial transcriptomics (10x Visium mouse brain)

*Scalable, likelihood-free estimation of a **Gaussian-process latent variable model** by centered
$Z_q$ estimating equations, applied to real spatial data.*

## What this is

We fit a **GP-GLLVM** to a 10x Visium mouse-brain section: $n\approx 2700$ spots, $p=100$
spatially-variable genes, integer counts. The model is

$$\varepsilon_k \sim \mathcal N(0, I),\quad z_{\cdot k} = L(\ell_k)\,\varepsilon_k,\quad
y_{ig}\mid z \sim \mathrm{Poisson}\!\big(\exp(w_g^\top z_i + b_g + o_i)\big),$$

where each latent factor $k$ is a **Gaussian process over the 2D spatial coordinates** with its own
isotropic length-scale $\ell_k$ (RBF kernel, independent latents $B=I$ — the identified GPFA
estimand), $W$ are the gene loadings, and $o_i=\log(\text{library size}_i/\text{median})$ is a
fixed per-spot offset so the factors capture spatial structure rather than sequencing depth.

## Why this matters (our thoughts)

- **The estimand nobody recovers for a GLLVM: per-factor spatial/temporal length-scales.** Standard
  GLLVM software (`gllvm`, VA/Laplace) has no GP latent; GP factor analysis (GPFA, MEFISTO,
  SpatialPCA) is Gaussian and does not scale to large non-Gaussian data. We estimate the
  $\ell_k$ *and* the loadings, likelihood-free, on raw counts.
- **It scales by construction.** Estimation only ever touches a local $K\times K$ patch of spots
  (the GP marginal theorem: any subset of a GP is itself an exact GP). Per-step cost is
  $O(\text{batch}\cdot K\cdot p + K^3)$ — **independent of the number of spots $n$**. So the same
  code runs on a Visium section or a Stereo-seq embryo with millions of bins.
- **The $Z_q$ recipe.** Loadings come from the centered data-anchored moment
  $-(m_1-m_2)$, $m=\mathbb E[\,\log(1{+}y)\cdot\eta\,]$ (real minus model-fantasy, the centering that
  makes the population root correct regardless of the encoder). The length-scales come from a
  **second-moment match** $\|\Sigma(\ell)-\hat\Sigma\|^2$ with
  $\hat\Sigma=\Sigma(\ell)+(\,\mathbb E_{\rm data}[\hat z\hat z^\top]-\mathbb E_{\rm fant}[\hat z\hat z^\top]\,)$
  — the fantasy-centered lag-covariance moment. *(This second-moment construction is what makes
  distinct per-factor $\ell_k$ separate; a per-observation $T(y)\!\cdot\!\eta$ moment alone cannot
  see the kernel.)*
- **Encoder = no-grad block-MAP imputation** over the patch (whitened by $L(\ell)^{-1}$); detached,
  so $\ell$ still gets a gradient through the decoder. By the score identity, the encoder choice
  affects only variance, not the population root.""")

co(r"""import sys, time, numpy as np, torch, matplotlib.pyplot as plt
from scipy.spatial import cKDTree
torch.set_default_dtype(torch.float64)
DATA = "data/visium_mouse_brain.npz"   # cached, preprocessed; regen recipe below
Q, K, MPOOL, JIT = 5, 80, 600, 1e-4    # factors, fit-patch size, wide pool, kernel jitter""")

md(r"""## Data

Cached in [`data/visium_mouse_brain.npz`](data/visium_mouse_brain.npz): raw integer counts `Y`
$(n\times p)$ for $p=100$ highly-variable genes (mitochondrial / `Bc1` dropped), 2D coordinates
`xy` rescaled so **1 unit = the nearest-neighbour spot pitch $\approx 100\,\mu$m**, and the
per-spot log-library-size `offset`. The regeneration recipe (run only if the cache is missing):

```python
import scanpy as sc
ad = sc.datasets.visium_sge(sample_id="V1_Mouse_Brain_Sagittal_Anterior"); ad.var_names_make_unique()
sc.pp.filter_genes(ad, min_cells=int(0.10*ad.n_obs))
ad = ad[:, ~(ad.var_names.str.startswith(("mt-","Mt-")) | (ad.var_names=="Bc1"))].copy()
lib = np.asarray(ad.X.sum(1)).ravel(); offset = np.log(lib/np.median(lib))
norm = ad.copy(); sc.pp.normalize_total(norm, target_sum=1e4); sc.pp.log1p(norm)
sc.pp.highly_variable_genes(norm, n_top_genes=100, flavor="seurat")
# Y = raw counts of HVGs; xy = (coords-mean)/pitch ; save Y, xy, offset, genes, pitch
```""")

co(r"""d = np.load(DATA, allow_pickle=True)
Y = torch.tensor(d["Y"]); XY = torch.tensor(d["xy"]); OFF = torch.tensor(d["offset"])
pitch, genes = float(d["pitch"]), list(d["genes"])
N, P = Y.shape
UM = 100 * pitch / 137.0          # 1 spot-unit in microns (Visium pitch ~100um)
print(f"{N} spots x {P} genes | coords {XY.min():.0f}..{XY.max():.0f} spot-units"
      f" | mean count/gene/spot {Y.mean():.2f} | 1 unit~{UM:.0f}um")""")

md(r"""## Model — 2D core (offset-aware, batched over patches)

Everything below is generic in the patch coordinates `c` (here 2D), so the same code is temporal,
spatial, or higher-dimensional. `Sigma_b` builds the block-diagonal $\Sigma(\ell)$ ($B=I$);
`encode_b` is the block-MAP imputer; `sample_y_b` draws model fantasies for the centering terms.""")

co(r"""def vec(Z):  return Z.transpose(1, 2).reshape(Z.shape[0], -1)            # (n,k,Q)->(n,Qk)
def unvec(u, k): return u.reshape(u.shape[0], Q, k).transpose(1, 2)         # (n,Qk)->(n,k,Q)

def Sigma_b(c, ells):                          # c:(n,k,2) -> (n,Qk,Qk) block-diag (B=I)
    n, k = c.shape[:2]
    d2 = ((c[:, :, None, :] - c[:, None, :, :]) ** 2).sum(-1)
    S = c.new_zeros(n, Q * k, Q * k)
    for q in range(Q):
        S[:, q*k:(q+1)*k, q*k:(q+1)*k] = torch.exp(-0.5 * d2 / ells[q] ** 2)
    return S + JIT * torch.eye(Q * k)

def sample_y_b(eps, c, off, W, b, ells):       # eps:(n,k,Q), off:(n,k) -> Poisson counts
    n, k, _ = eps.shape
    z = unvec((torch.linalg.cholesky(Sigma_b(c, ells)) @ vec(eps).unsqueeze(-1)).squeeze(-1), k)
    eta = torch.einsum("nkq,pq->nkp", z, W) + b + off[..., None]
    return torch.poisson(torch.exp(eta.clamp(max=12)))

def encode_b(y, c, off, W, b, ells, s2=1.0):   # block-MAP imputer -> zhat (n,k,Q)
    n, k, _ = y.shape
    L = torch.linalg.cholesky(Sigma_b(c, ells))
    Lit = torch.linalg.solve_triangular(L, torch.eye(Q*k).expand(n, -1, -1), upper=False)
    A = Lit.transpose(1, 2) @ Lit + torch.kron(W.T @ W / s2, torch.eye(k))
    rhs = vec((torch.log1p(y) - b - off[..., None]) @ W / s2)
    return unvec(torch.cholesky_solve(rhs.unsqueeze(-1), torch.linalg.cholesky(A)).squeeze(-1), k)

# patches: dense KNN (for imputation) + a wide pool we draw K from (so distances span the scale)
tree = cKDTree(d["xy"])
idx_dense = torch.tensor(tree.query(d["xy"], k=64)[1])
idx_pool  = torch.tensor(tree.query(d["xy"], k=MPOOL)[1])
pool_R = float(np.sqrt(MPOOL / np.pi))
def patch(batch):
    pool = idx_pool[torch.randint(0, N, (batch,))]
    sel = torch.rand(batch, MPOOL).argsort(1)[:, :K]
    ii = pool.gather(1, sel)
    return Y[ii], XY[ii], OFF[ii]
print(f"fit patch K={K} drawn from {MPOOL}-NN (radius~{pool_R:.0f} units ~ {pool_R*UM:.0f}um)")""")

md(r"""## Fit

Adam on $(W,b,\log\ell)$; loadings via the centered $-(m_1-m_2)$, length-scales via the
fantasy-centered second-moment match (warm-up holds $\ell$ fixed first). **Why a *wide* pool:**
to estimate $\ell$ the patch must contain pairs at distances $\sim\ell$ — a dense KNN patch
(radius ~4 units) can only see ~400 µm and the $\ell$ saturate at that ceiling. Drawing $K$ spots
sparsely from a wide neighbourhood lets the same $K\times K$ cost span ~1.4 mm.""")

co(r"""def fit(steps=600, lr=0.03, batch=48, warm=200, seed=0):
    torch.manual_seed(seed)
    W = torch.nn.Parameter(torch.randn(P, Q) * 0.3)
    b = torch.nn.Parameter(torch.log1p(Y.mean(0)))
    log_ell = torch.nn.Parameter(torch.log(torch.linspace(1.0, 12.0, Q)))
    opt = torch.optim.Adam([W, b, log_ell], lr=lr); trace = []
    for it in range(steps):
        ells = log_ell.exp(); yb, cb, ob = patch(batch)
        with torch.no_grad():
            Wd, bd, eld = W.detach(), b.detach(), ells.detach()
            yq = sample_y_b(torch.randn(batch, K, Q), cb, ob, Wd, bd, eld)
            zd = encode_b(yb, cb, ob, Wd, bd, eld); zq = encode_b(yq, cb, ob, Wd, bd, eld)
        ed = torch.einsum("nkq,pq->nkp", zd, W) + b + ob[..., None]
        eq = torch.einsum("nkq,pq->nkp", zq, W) + b + ob[..., None]
        loss = -((torch.log1p(yb) * ed).sum(-1).mean() - (torch.log1p(yq) * eq).sum(-1).mean())
        with torch.no_grad():
            Md = torch.einsum("ni,nj->ij", vec(zd), vec(zd)) / batch
            Mq = torch.einsum("ni,nj->ij", vec(zq), vec(zq)) / batch
            Sig_hat = Sigma_b(cb[:1], ells)[0] + (Md - Mq)
        loss = loss + ((Sigma_b(cb[:1], ells)[0] - Sig_hat) ** 2).mean()
        opt.zero_grad(); loss.backward()
        if it < warm: log_ell.grad = None
        torch.nn.utils.clip_grad_norm_([W, b, log_ell], 5.0); opt.step()
        if it % 50 == 0: trace.append((it, *sorted(log_ell.exp().tolist())))
    return W.detach(), b.detach(), log_ell.exp().detach(), trace

t0 = time.time(); W, b, ells, trace = fit(); fit_t = time.time() - t0
print(f"fit {fit_t:.0f}s on {N} spots (cost independent of n)")
for q in np.argsort(ells.numpy()):
    print(f"  factor {q}:  ell = {ells[q]:.2f} spot-units  =  {ells[q]*UM:.0f} um")""")

md(r"""## Factor maps + length-scales

Impute a latent map by encoding each spot's dense local patch and keeping the centre — chunked,
so it stays $O(n)$. Factors are ordered fine $\to$ coarse; visually the smoothness tracks $\ell$.""")

co(r"""zmap = torch.empty(N, Q)
for s in range(0, N, 256):
    ii = idx_dense[s:s+256]
    zmap[s:s+256] = encode_b(Y[ii], XY[ii], OFF[ii], W, b, ells)[:, 0, :]

xy = d["xy"]; order = np.argsort(ells.numpy())
fig, ax = plt.subplots(1, Q, figsize=(3.1 * Q, 3.4))
for j, q in enumerate(order):
    z = zmap[:, q].numpy(); v = np.percentile(np.abs(z), 98)
    ax[j].scatter(xy[:, 0], xy[:, 1], c=z, s=6, cmap="RdBu_r", vmin=-v, vmax=v)
    ax[j].set_title(f"factor {q}   $\\ell$={ells[q]*UM:.0f}$\\mu$m"); ax[j].set_aspect("equal")
    ax[j].axis("off"); ax[j].invert_yaxis()
fig.suptitle(f"GP-GLLVM on Visium mouse brain ({N} spots, {P} genes) — fit {fit_t:.0f}s, local K={K} patches", y=1.03)
fig.tight_layout(); plt.show()""")

md(r"""## What we see, and honest caveats

**Result.** The factors recover **distinct per-factor spatial length-scales** spanning roughly
spot-resolution to ~1.5 mm, and the maps are spatially coherent anatomical patterns whose
smoothness tracks the estimated $\ell$. This is the headline deliverable: *per-factor spatial
timescales for a GLLVM, on raw counts, at a cost independent of the number of spots.*

**Caveats (kept honest):**
1. **Coarsest $\ell$ sits near the fit-pool radius (~14 units).** It is likely ceiling-limited and
   could be larger; widening the pool would confirm whether it stabilises or keeps climbing.
2. **Validation here is qualitative** (smoothness $\leftrightarrow\ell$, anatomical coherence).
   There is no ground-truth $\ell$ on real data; the quantitative check for the paper is a
   **parametric bootstrap** — simulate from the fitted model (frozen surrogate), re-estimate, and
   confirm $\ell$ recovery and coverage. The $Z_q$ theory (frozen-surrogate bootstrap) is built
   for exactly this.
3. The dense-vs-wide patch split is a pragmatic choice, not an optimised sampler.

**Positioning.** Closest relatives are **GPFA** (Yu 2009, temporal, Gaussian), **MEFISTO**
(GP-structured factors, Gaussian/MOFA), and **SpatialPCA** — all Gaussian and $O(n^3)$-bound. Our
edge is **non-Gaussian (Poisson) + likelihood-free + flat cost in $n$** via the $Z_q$ centered
equations on GP marginal subsets. In the paper this is the GP-GLLVM extension and the real-data
scalability demonstration (cf. `paper/` §Method/§Extension and `simulation_5/`).""")

nb["cells"] = C
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
out = "/home/willwhite/GitHub/gllvm/simulations/simulation_5/visium_gp_gllvm.ipynb"
nbf.write(nb, out); print("wrote", out)
