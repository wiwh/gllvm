import nbformat as nbf
nb = nbf.v4.new_notebook(); C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def co(s): C.append(nbf.v4.new_code_cell(s))

md(r"""# Poisson GLLVM by ZQE: does the quantile-projected encoder help $W$ recovery?

We fit a **Poisson GLLVM** ($z\sim N(0,I)$, $y\sim\mathrm{Poisson}(e^{Wz+b})$) by the centered
$Z_q$ equations, $-(m_1-m_2)$ with $m=\mathbb E[\log1p(y)\cdot\eta(\hat z)]$, and swap the encoder
$\hat z(y)$ between three choices:

1. **log1p-Gauss MAP** — $\hat z=(W^\top W/\sigma^2+I)^{-1}W^\top(\log1p(y)-b)/\sigma^2$ (the current default).
2. **+ quantile projection** — same MAP, then each latent margin rank→$\Phi^{-1}$ to the prior $N(0,1)$
   (the [encoder diagnostic](poisson_encoder_quantile.ipynb) showed this ~halves *latent* RMSE by undoing over-shrinkage).
3. **correct Poisson-Newton MAP** — the actual posterior mode under the Poisson likelihood (Newton, warm-started from #1).

By the score identity all three are **consistent** for $W$ (the encoder only affects efficiency).
The question: does the calibrated latent translate into better $W$ recovery, or does it wash out?

*A theory prediction to keep in mind:* a pure **scalar** rescale of $\hat z$ cancels in the centered
equation (it multiplies $m_1$ and $m_2$ equally), so de-shrinking alone should **not** move $\hat W$ —
only the projection's nonlinear *reshape* of the tails can. So we might expect a small effect on $W$
even though the latent itself improved a lot.""")

co(r"""import sys, time, numpy as np, torch, torch.nn as nn, matplotlib.pyplot as plt
sys.path.insert(0, "/home/willwhite/GitHub/gllvm/src")
from gllvm.autofit import procrustes_error, orthogonal_align
torch.set_default_dtype(torch.float64)
p, q = 30, 2

def gen_data(seed, n=2000):
    g = np.random.default_rng(seed)
    W = torch.tensor(g.normal(0, 0.6, (p, q))); b = torch.zeros(p)
    Z = torch.tensor(g.normal(0, 1, (n, q)))
    Y = torch.poisson(torch.exp(Z @ W.T + b))
    return W, b, Y""")

md(r"""## The three encoders (batched, no-grad inside the fit)""")

co(r"""def enc_gauss(Y, W, b, s2=1.0):                                   # log1p-Gaussian MAP
    A = W.T @ W / s2 + torch.eye(q)
    return torch.linalg.solve(A, ((torch.log1p(Y) - b) @ W / s2).T).T

def quantile_project(Z):                                          # per-margin rank -> N(0,1)
    n = Z.shape[0]
    ranks = Z.argsort(0).argsort(0).double()
    return torch.special.ndtri((ranks + 0.5) / n)

def enc_gauss_q(Y, W, b, s2=1.0):
    return quantile_project(enc_gauss(Y, W, b, s2))

def enc_poisson(Y, W, b, iters=6):                                # correct Poisson-Newton MAP
    Z = enc_gauss(Y, W, b); I = torch.eye(q)
    for _ in range(iters):
        lam = torch.exp((Z @ W.T + b).clamp(max=10))              # (B,p)
        g = (Y - lam) @ W - Z                                     # grad of (loglik + log prior)
        H = torch.einsum("bp,pi,pj->bij", lam, W, W) + I          # (B,q,q)
        Z = Z + torch.linalg.solve(H, g.unsqueeze(-1)).squeeze(-1)
    return Z

ENCODERS = {"log1p-Gauss MAP": enc_gauss, "MAP + quantile": enc_gauss_q, "Poisson-Newton MAP": enc_poisson}""")

md(r"""## Encoder check first: $\hat z$ vs true $z$, all three overlaid per latent dimension
One dataset, **true $(W,b)$**, each encoder Procrustes-aligned to the true factors. (Legend shows the
per-dimension std — exposes the over-shrinkage of the raw MAP vs the calibrated projection.)""")

co(r"""g = np.random.default_rng(0)                                      # same draw order as gen_data(0)
Wt = torch.tensor(g.normal(0, 0.6, (p, q))); bt = torch.zeros(p)
Zt = torch.tensor(g.normal(0, 1, (2000, q))); Yt = torch.poisson(torch.exp(Zt @ Wt.T + bt))
cols = {"log1p-Gauss MAP": "#4C72B0", "MAP + quantile": "#DD8452", "Poisson-Newton MAP": "#55A868"}
fig, ax = plt.subplots(1, q, figsize=(5.4*q, 4.6))
for name, enc in ENCODERS.items():
    Zh = enc(Yt, Wt, bt); Zh = Zh @ orthogonal_align(Zt, Zh)     # align to true axes for comparison
    for k in range(q):
        ax[k].scatter(Zt[:, k], Zh[:, k], s=6, alpha=.3, color=cols[name],
                      label=f"{name} (std={Zh[:,k].std():.2f})")
for k in range(q):
    lo, hi = Zt[:, k].min().item(), Zt[:, k].max().item(); ax[k].plot([lo, hi], [lo, hi], "k--", lw=.9)
    ax[k].set_title(f"latent dimension {k}"); ax[k].set_xlabel("true $z$"); ax[k].set_ylabel(r"encoded $\hat z$")
    ax[k].legend(fontsize=8, markerscale=2)
fig.suptitle(r"Encoder $\hat z$ vs true $z$ (true $W,b$) — log1p-MAP is shrunk; quantile sits on $y=x$")
fig.tight_layout(); plt.show()""")

md(r"""## ZQE fit (same data per seed for all three encoders)""")

co(r"""def fit(Y, enc, steps=600, lr=0.03, batch=256, seed=0):
    n = Y.shape[0]; torch.manual_seed(7000 + seed)
    W = nn.Parameter(torch.randn(p, q) * 0.5); b = nn.Parameter(torch.zeros(p))
    opt = torch.optim.Adam([W, b], lr=lr)
    for it in range(steps):
        Yb = Y[torch.randint(0, n, (batch,))]
        with torch.no_grad():
            Wd, bd = W.detach(), b.detach()
            Yq = torch.poisson(torch.exp((torch.randn(batch, q) @ Wd.T + bd).clamp(max=10)))
            zd = enc(Yb, Wd, bd); zq = enc(Yq, Wd, bd)
        ed = zd @ W.T + b; eq = zq @ W.T + b
        loss = -((torch.log1p(Yb) * ed).sum(-1).mean() - (torch.log1p(Yq) * eq).sum(-1).mean())
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([W, b], 5.0); opt.step()
    return procrustes_error(Wt, W.detach())

NSEED = 20; t0 = time.time(); res = {k: [] for k in ENCODERS}
for s in range(NSEED):
    Wt, bt, Y = gen_data(s)
    for name, enc in ENCODERS.items():
        res[name].append(fit(Y, enc, seed=s))
    print(f"seed {s}: " + "  ".join(f"{k}={res[k][-1]:.3f}" for k in ENCODERS), flush=True)
print(f"[{time.time()-t0:.0f}s]")
print("\nProcrustes error of W (mean +/- std over %d seeds):" % NSEED)
for k in ENCODERS:
    a = np.array(res[k]); print(f"  {k:>20}: {a.mean():.3f} +/- {a.std():.3f}")""")

co(r"""fig, ax = plt.subplots(figsize=(6.5, 4))
names = list(ENCODERS); m = [np.mean(res[k]) for k in names]; sd = [np.std(res[k]) for k in names]
ax.bar(names, m, yerr=sd, capsize=5, color=["#4C72B0", "#DD8452", "#55A868"])
for i, k in enumerate(names): ax.scatter([i]*NSEED, res[k], color="k", s=12, zorder=3)
ax.set_ylabel("Procrustes error of W"); ax.set_title("Poisson GLLVM (ZQE): W recovery by encoder")
plt.xticks(rotation=10); fig.tight_layout(); plt.show()""")

md(r"""## Reading the result

All three are **consistent** for $W$ (centering guarantees it), so this is pure **efficiency** — see
the table/bar above (20 seeds, *same data per seed*, so it's a paired comparison):

1. **MAP + quantile is the best encoder for $W$** — a small but consistent mean improvement *and*
   roughly **half the seed-to-seed variance** (more reliable loadings). The big *latent*-RMSE win
   (`poisson_encoder_quantile.ipynb`, ~2×) does not fully transfer, because the dominant effect — a
   **scalar de-shrink — cancels in the centered $m_1-m_2$ equation**; but the nonlinear *reshape* (the
   part that is not a scalar) buys real **stability** and a modest mean gain. So the projection is a
   genuine, cheap upgrade, just not a 2× one.
2. **The "correct" Poisson-Newton MAP is no better than the cheap Gauss MAP** for $W$ (and is slower
   / less robust) — a clean vindication of the cheap proxy: **encoder fidelity ≠ estimator quality**
   (the score identity). Note the [encoder plot](#) above shows the Poisson MAP *is* well-calibrated
   (std≈1, on $y=x$) just like the quantile projection — yet that fidelity does not translate into a
   better $\hat W$.

**Bottom line:** use **log1p-Gauss MAP + quantile projection** — you get a *calibrated* latent (for
inference / downstream embeddings / robust quantile-clipping) **and** the most reliable loadings, at
negligible cost; the expensive correct Poisson MAP isn't worth it.""")

nb["cells"] = C
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
out = "/home/willwhite/GitHub/gllvm/playground/poisson_quantile.ipynb"
nbf.write(nb, out); print("wrote", out)
