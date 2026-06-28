"""
simulation_8 — standard errors for ZQE loadings via a frozen-encoder parametric bootstrap.

Idea (user's): once $\\hat\\theta$ is fit, **freeze the encoder** (the score identity says its
$\\theta$-dependence is first-order free near the solution). Then a standard *parametric
bootstrap* of the estimator is, per replicate, **just a GLM**:

    1. simulate a fantasy dataset  Ytil ~ f_{theta_hat}              (n x p counts)
    2. encode it with the FROZEN encoder  ztil = MAP(Ytil; theta_hat) (n x q)
    3. refit the decoder with z held fixed  ->  p independent Poisson GLMs
       (regress each response on [ztil | 1]; batched Newton)          -> (W^(h), b^(h))

Repeat H times; the spread of {W^(h)} is the sampling variance of the estimator. A parametric
bootstrap is **self-calibrating** — there is no factor-of-2 to get right (that ambiguity only
arises in the alternative "re-solve the centered estimating equation with one fantasy" route;
see paper/consistency notes). The only approximation is freezing the encoder, validated here by
coverage.

We report standard errors / Wald CIs and **validate by coverage** on the rotation-invariant Gram
entries $(WW^\\top)_{jk}=w_j^\\top w_k$ (no gauge bookkeeping needed: orthogonal rotations cancel).
"""
from __future__ import annotations

import copy
import time
import numpy as np
import torch

from gllvm.gllvm_module import GLLVM
from gllvm.glms import PoissonGLM
from gllvm.encoder import MapEncoderPoissonNewton
from gllvm.autofit import ZQEAutoFitter, procrustes_error
from gllvm.glm_fit import poisson_newton_batch, initial_gaussian_fit
from gllvm.simulations import make_sparse, simulate

ZQE_KW = dict(steps_per_round=150, max_rounds=2, tol=0.001,
              warmup_lr=0.5, refine_lr=0.5, ema_decay=0.95, verbose=False)


# ----------------------------------------------------------------------------
# data / point fit (same setup as poisson.ipynb / sim_7: sparse, lower_tri)
# ----------------------------------------------------------------------------
def make_truth(p, q, *, wz_scale=0.5, seed=0, lower_tri=True, responses_per_latent=None):
    """Draw a fixed true GLLVM (the loadings). Returns the GLLVM object."""
    torch.manual_seed(seed)
    rpl = p // 2 if responses_per_latent is None else responses_per_latent
    return make_sparse(n_latent=q, poisson=p, active_latent=q, wz_scale=wz_scale,
                       responses_per_latent=rpl, lower_tri=lower_tri)


def sample_data(g_true, n, *, seed=0):
    """Draw one dataset (fresh z + noise) from a FIXED true model."""
    torch.manual_seed(seed)
    Y, _ = simulate(g_true, n_samples=n, device="cpu")
    return Y


def make_data(p, q, n, *, wz_scale=0.5, seed=0, lower_tri=True, responses_per_latent=None):
    """Convenience: a truth + one dataset from it (truth and data share ``seed``)."""
    g = make_truth(p, q, wz_scale=wz_scale, seed=seed, lower_tri=lower_tri,
                   responses_per_latent=responses_per_latent)
    Y = sample_data(g, n, seed=seed)
    return Y, g.wz.detach().cpu().clone(), g.bias.detach().cpu().clone()


def fit_point(Y, q, *, l2, device, seed=0, wz_scale=0.5, lower_tri=True, **kw):
    p = Y.shape[1]
    torch.manual_seed(seed)
    g = GLLVM(latent_dim=q, output_dim=p, bias=True, lower_tri=lower_tri).to(device)
    g.add_glm(PoissonGLM, idx=list(range(p)), params={"T": torch.log1p}, name="P")
    with torch.no_grad():
        torch.nn.init.normal_(g.wz, std=wz_scale)
        g.bias.zero_()
    kwargs = {**ZQE_KW, **kw}
    ft = ZQEAutoFitter(
        g, encoder_factory=lambda g: MapEncoderPoissonNewton(g, lam=1.0, max_iter=30),
        device=device, seed=seed, l2=l2, **kwargs,
    ).fit(Y.to(device))
    return ft


# ----------------------------------------------------------------------------
# the "GLM per draw" refit:  solve for (W, b) with z held fixed
# ----------------------------------------------------------------------------
def refit_glm(z, Ycounts, *, lam=0.0, max_iter=50):
    """p independent Poisson GLMs: regress each response (column of Ycounts) on [z | 1].

    Uses the batched Newton solver. ``z`` is (n, q), ``Ycounts`` is (n, p);
    returns ``(W (p, q), b (p,))``.
    """
    n, q = z.shape
    X = torch.cat([z, torch.ones(n, 1, device=z.device, dtype=z.dtype)], dim=1)  # (n, q+1)
    Yt = Ycounts.to(X.dtype)                                                      # (n, p)
    B0 = initial_gaussian_fit(X, Yt)                                              # (q+1, p)
    B, _ = poisson_newton_batch(X=X, Y=Yt, B0=B0, lam=lam, max_iter=max_iter, verbose=False)
    return B[:q].T.contiguous(), B[q].contiguous()


# ----------------------------------------------------------------------------
# frozen-encoder parametric bootstrap
# ----------------------------------------------------------------------------
def param_bootstrap(ft, H, *, device, seed=0, lam_enc=1.0, lam_glm=0.0, clamp=10.0):
    """[SUPERSEDED — wrong statistic] Frozen-encoder bootstrap whose decoder refit is a canonical
    Poisson GLM (T=Y). That is the MORE-efficient statistic, so its SE is too small for the
    T=log1p estimator -> CIs too narrow -> undercoverage. The correct bootstrap re-solves the
    estimator's OWN equations (T=log1p) with a fixed (CRN) centering: ``param_bootstrap_resolve``.
    Kept only as the explicit T=Y contrast."""
    g = ft.model
    What = g.wz.detach().to(device)
    bhat = g.bias.detach().to(device)
    p, q = What.shape
    n = ft.y_.shape[0]
    enc = MapEncoderPoissonNewton(g, lam=lam_enc, max_iter=30)  # live ref -> frozen at theta_hat
    gen = torch.Generator(device=device).manual_seed(seed)

    W_boot = torch.empty(H, p, q)
    for h in range(H):
        zsim = torch.randn(n, q, generator=gen, device=device)
        rate = torch.exp((zsim @ What.T + bhat).clamp(max=clamp))     # (n, p)
        Ytil = torch.poisson(rate, generator=gen)
        ztil, _, _ = enc.sample(Ytil)                                 # frozen-encoder codes
        Wh, _ = refit_glm(ztil, Ytil, lam=lam_glm)                    # canonical T=Y (wrong here)
        W_boot[h] = Wh.cpu()
    return W_boot


def fixed_point_solve(g_hat, Y, *, device, fantasy_seed=0, tol=1e-3, max_iter=30,
                      lam_glm=0.0, clamp=12.0, return_iters=False):
    """CRN bias-corrected GLM fixed point, run to **FULL CONVERGENCE** (warm-started at theta_hat;
    converging fully removes any dependence on the warm start -> no compression of the bootstrap
    spread). Live Poisson-MAP encoder (parameter-free, tracks theta). Full fantasy seed kept
    (zsim + Yq both CRN), so the fantasy is a deterministic function of theta and moves smoothly.
    Iterates until the gauge-invariant change procrustes_error(W_prev, W_new) < tol. Each iter is
    two batched Poisson GLMs:

        z_d = PoissonMAP(Y; theta);   theta_d = GLM(Y on z_d)
        zsim,Yq = CRN(seed);          z_q = PoissonMAP(Y_q; theta);  theta_q = GLM(Y_q on z_q)
        theta <- theta_d - (theta_q - theta)        # centering = bias correction

    Returns (W, b) (or (W, b, n_iters) if return_iters)."""
    Y = Y.to(device)
    n, p = Y.shape
    g_work = copy.deepcopy(g_hat).to(device)
    enc = MapEncoderPoissonNewton(g_work, lam=1.0, max_iter=30)   # parameter-free; reads g_work live
    W = g_hat.wz.detach().to(device).clone(); b = g_hat.bias.detach().to(device).clone()
    q = W.shape[1]
    it = 0
    for it in range(1, max_iter + 1):
        with torch.no_grad():
            g_work.wz.copy_(W); g_work.bias.copy_(b)             # encoder tracks current theta
            z_d, _, _ = enc.sample(Y)                            # PoissonMAP(data; theta)
            gz = torch.Generator(device=device).manual_seed(fantasy_seed)
            zsim = torch.randn(n, q, generator=gz, device=device)        # CRN latent
            rate = torch.exp((zsim @ W.T + b).clamp(max=clamp))          # moves with theta
            gp = torch.Generator(device=device).manual_seed(fantasy_seed + 1)
            Yq = torch.poisson(rate, generator=gp)                       # CRN counts
            z_q, _, _ = enc.sample(Yq)                           # PoissonMAP(fantasy; theta)
        W_d, b_d = refit_glm(z_d, Y, lam=lam_glm)
        W_q, b_q = refit_glm(z_q, Yq, lam=lam_glm)
        W_new = W_d - (W_q - W); b_new = b_d - (b_q - b)         # bias correction (centering)
        change = procrustes_error(W.detach().cpu(), W_new.detach().cpu())  # gauge-invariant
        W, b = W_new, b_new
        if change < tol:
            break
    return (W, b, it) if return_iters else (W, b)


def param_bootstrap_crn(ft, H, *, device, seed=0, fantasy_seed=12345, tol=1e-3, max_iter=30,
                        lam_glm=0.0, clamp=12.0):
    """Corrected parametric bootstrap: per replicate simulate Ytil ~ f_{theta_hat} (the data
    variance source), then re-solve the CRN bias-corrected GLM fixed point to **full convergence**
    (warm-started at theta_hat -> converged, so no warm-start compression). ``fantasy_seed`` is
    the SAME across replicates, so the fantasy is common and does not inflate the spread; only the
    bootstrap data varies. Returns W_boot (H, p, q) on CPU."""
    g = ft.model
    What = g.wz.detach().to(device); bhat = g.bias.detach().to(device)
    p, q = What.shape; n = ft.y_.shape[0]
    gen = torch.Generator(device=device).manual_seed(seed)
    W_boot = torch.empty(H, p, q)
    for h in range(H):
        zb = torch.randn(n, q, generator=gen, device=device)
        Ytil = torch.poisson(torch.exp((zb @ What.T + bhat).clamp(max=clamp)))
        Wh, _ = fixed_point_solve(g, Ytil, device=device, fantasy_seed=fantasy_seed,
                                  tol=tol, max_iter=max_iter, lam_glm=lam_glm, clamp=clamp)
        W_boot[h] = Wh.cpu()
    return W_boot


class _FtShim:
    """Minimal stand-in for a fitter (just .model and .y_) so param_bootstrap_crn can bootstrap
    around an arbitrary theta (e.g. the converged fixed-point estimate)."""
    def __init__(self, model, y_):
        self.model = model; self.y_ = y_


def _gllvm_at(W, b, *, lower_tri=True):
    p, q = W.shape
    g = GLLVM(latent_dim=q, output_dim=p, bias=True, lower_tri=lower_tri)
    with torch.no_grad():
        g.wz.copy_(W); g.bias.copy_(b)
    return g


def param_bootstrap_resolve(ft, H, *, device, l2, seed=0, fantasy_seed=12345, clamp=12.0,
                            lower_tri=True, **zqe_over):
    """Cleanest CRN bootstrap: re-solve the **actual estimator** (ZQEAutoFitter, Poisson-MAP,
    T=log1p) on each simulated dataset, **warm-started at theta_hat** (``warmup_max_epochs=0`` ->
    no random-init warmup) with a **fixed seed** (common random numbers, so the fantasy MC noise
    is common across replicates and cancels; only the bootstrap data varies). This is precisely
    theta_hat's own estimator with a frozen seed -> the bootstrap spread is Var(theta_hat).
    Returns W_boot (H, p, q) on CPU."""
    g = ft.model
    What = g.wz.detach().to(device); bhat = g.bias.detach().to(device)
    p, q = What.shape; n = ft.y_.shape[0]
    gen = torch.Generator(device=device).manual_seed(seed)
    kw = dict(steps_per_round=150, max_rounds=2, tol=1e-3, warmup_lr=0.5, refine_lr=0.5,
              ema_decay=0.95, verbose=False, warmup_max_epochs=0)   # no warmup -> warm-start at theta_hat
    kw.update(zqe_over)
    W_boot = torch.empty(H, p, q)
    for h in range(H):
        zb = torch.randn(n, q, generator=gen, device=device)
        Ytil = torch.poisson(torch.exp((zb @ What.T + bhat).clamp(max=clamp)))
        gb = GLLVM(latent_dim=q, output_dim=p, bias=True, lower_tri=lower_tri).to(device)
        gb.add_glm(PoissonGLM, idx=list(range(p)), params={"T": torch.log1p}, name="P")
        with torch.no_grad():
            gb.wz.copy_(What); gb.bias.copy_(bhat)             # warm-start at theta_hat
        ftb = ZQEAutoFitter(
            gb, encoder_factory=lambda g: MapEncoderPoissonNewton(g, lam=1.0, max_iter=30),
            device=device, seed=fantasy_seed, l2=l2, **kw,      # fixed seed = CRN across replicates
        ).fit(Ytil)
        W_boot[h] = ftb.model.wz.detach().cpu()
    return W_boot


def gram_entries(W, idx):
    """Rotation-invariant functionals: (W W^T)_{jk} = w_j . w_k for the (j,k) pairs in ``idx``.
    ``W`` is (..., p, q); ``idx`` is a long tensor (m, 2). Returns (..., m)."""
    j, k = idx[:, 0], idx[:, 1]
    return (W[..., j, :] * W[..., k, :]).sum(-1)


def select_gram_idx(p, q, n_offdiag=200, seed=0):
    """All diagonal (j,j) + a random sample of off-diagonal (j,k), j<k. (m, 2) long tensor."""
    g = torch.Generator().manual_seed(seed)
    diag = torch.stack([torch.arange(p), torch.arange(p)], 1)
    m = min(n_offdiag, p * (p - 1) // 2)
    js = torch.randint(0, p, (4 * m,), generator=g)
    ks = torch.randint(0, p, (4 * m,), generator=g)
    keep = js < ks
    off = torch.stack([js[keep], ks[keep]], 1)[:m]
    return torch.cat([diag, off], 0)


# ----------------------------------------------------------------------------
# coverage experiment: D datasets from a fixed truth -> empirical Wald coverage
# ----------------------------------------------------------------------------
def coverage_experiment_crn(p, q, n, *, D=40, H=200, wz_scale=0.5, l2_coef=0.001,
                            device=None, level=0.95, truth_seed=0, tol=1e-3, max_iter=30,
                            fantasy_seed=999, verbose=True):
    """Self-consistent coverage for the **CRN fixed-point estimator** (T=y bias-corrected GLM,
    live Poisson-MAP encoder), run to FULL CONVERGENCE (warm-started, but converged -> no
    warm-start compression). Point estimate = converged fixed point; SE = CRN bootstrap around it.
    Reports plain, **bias-corrected** (2*ghat - mean(g*)), and basic/pivotal CIs (the estimator is
    expected off-center, so debiasing matters), plus the SE ratio (boot/emp)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    z_crit = float(torch.distributions.Normal(0, 1).icdf(torch.tensor(0.5 + level / 2)))
    alpha = 1.0 - level
    truth = make_truth(p, q, wz_scale=wz_scale, seed=truth_seed)
    W_true = truth.wz.detach().cpu().clone()
    idx = select_gram_idx(p, q, seed=truth_seed)
    g_true = gram_entries(W_true, idx)

    cov_w, cov_bc, cov_bas, ghat_all, se_all, recs = [], [], [], [], [], []
    for d in range(D):
        t0 = time.time()
        Y = sample_data(truth, n, seed=1000 + d)
        ft = fit_point(Y, q, l2=l2_coef / n, device=device, seed=1000 + d, wz_scale=wz_scale)
        W_fp, b_fp, n_it = fixed_point_solve(ft.model, Y, device=device, fantasy_seed=fantasy_seed,
                                             tol=tol, max_iter=max_iter, return_iters=True)
        g_fp = _gllvm_at(W_fp.cpu(), b_fp.cpu()).to(device)
        g_hat = gram_entries(W_fp.cpu(), idx)
        W_boot = param_bootstrap_crn(_FtShim(g_fp, Y), H, device=device, seed=7000 + d,
                                     fantasy_seed=fantasy_seed, tol=tol, max_iter=max_iter)
        g_boot = gram_entries(W_boot, idx)
        se = g_boot.std(0, unbiased=True)
        # plain Wald
        cov_w.append(((g_true >= g_hat - z_crit * se) & (g_true <= g_hat + z_crit * se)).float())
        # bias-corrected Wald (recenter: the fixed-point estimator sits off-center)
        g_bc = 2.0 * g_hat - g_boot.mean(0)
        cov_bc.append(((g_true >= g_bc - z_crit * se) & (g_true <= g_bc + z_crit * se)).float())
        # basic/pivotal
        qlo = torch.quantile(g_boot, alpha / 2, dim=0); qhi = torch.quantile(g_boot, 1 - alpha / 2, dim=0)
        cov_bas.append(((g_true >= 2 * g_hat - qhi) & (g_true <= 2 * g_hat - qlo)).float())
        ghat_all.append(g_hat); se_all.append(se)
        pw = procrustes_error(W_true, W_fp.cpu())
        recs.append(dict(d=d, procW=float(pw), n_iter=n_it, t=time.time() - t0))
        if verbose:
            print(f"[{d+1:>3}/{D}] procW={pw:.3f}  iters={n_it}  wald={cov_w[-1].mean():.3f}  "
                  f"bc={cov_bc[-1].mean():.3f}  basic={cov_bas[-1].mean():.3f}  "
                  f"({recs[-1]['t']:.1f}s)", flush=True)

    cov_w = torch.stack(cov_w); cov_bc = torch.stack(cov_bc); cov_bas = torch.stack(cov_bas)
    ghat_all = torch.stack(ghat_all); se_all = torch.stack(se_all)
    ratio = se_all.mean(0) / ghat_all.std(0, unbiased=True).clamp_min(1e-12)

    def _s(c):
        return dict(overall=float(c.mean()), diag=float(c[:, :p].mean()), offdiag=float(c[:, p:].mean()))
    return dict(
        nominal=level,
        wald=_s(cov_w), bias_corrected=_s(cov_bc), basic=_s(cov_bas),
        se_ratio=dict(overall=float(ratio.median()), diag=float(ratio[:p].median()),
                      offdiag=float(ratio[p:].median())),
        records=recs,
    )


def coverage_experiment_resolve(p, q, n, *, D=10, H=50, wz_scale=0.5, l2_coef=0.001,
                                device=None, level=0.95, truth_seed=0, fantasy_seed=999,
                                verbose=True):
    """Coverage for the CLEANEST variance estimator: point = theta_hat (ZQEAutoFitter, the actual
    method); SE = CRN bootstrap that **re-solves the same estimator** warm-started at theta_hat with
    a fixed seed (param_bootstrap_resolve). Self-consistent (point and bootstrap are the same
    estimator). Reports plain + bias-corrected Wald and the SE ratio (boot/emp)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    z_crit = float(torch.distributions.Normal(0, 1).icdf(torch.tensor(0.5 + level / 2)))
    truth = make_truth(p, q, wz_scale=wz_scale, seed=truth_seed)
    W_true = truth.wz.detach().cpu().clone()
    idx = select_gram_idx(p, q, seed=truth_seed)
    g_true = gram_entries(W_true, idx)

    cov_w, cov_bc, ghat_all, se_all, recs = [], [], [], [], []
    for d in range(D):
        t0 = time.time()
        Y = sample_data(truth, n, seed=1000 + d)
        ft = fit_point(Y, q, l2=l2_coef / n, device=device, seed=1000 + d, wz_scale=wz_scale)
        g_hat = gram_entries(ft.model.wz.detach().cpu(), idx)
        W_boot = param_bootstrap_resolve(ft, H, device=device, l2=l2_coef / n, seed=7000 + d,
                                         fantasy_seed=fantasy_seed)
        g_boot = gram_entries(W_boot, idx)
        se = g_boot.std(0, unbiased=True)
        cov_w.append(((g_true >= g_hat - z_crit * se) & (g_true <= g_hat + z_crit * se)).float())
        g_bc = 2.0 * g_hat - g_boot.mean(0)
        cov_bc.append(((g_true >= g_bc - z_crit * se) & (g_true <= g_bc + z_crit * se)).float())
        ghat_all.append(g_hat); se_all.append(se)
        pw = procrustes_error(W_true, ft.model.wz.detach().cpu())
        recs.append(dict(d=d, procW=float(pw), t=time.time() - t0))
        if verbose:
            print(f"[{d+1:>3}/{D}] procW={pw:.3f}  wald={cov_w[-1].mean():.3f}  "
                  f"bc={cov_bc[-1].mean():.3f}  ({recs[-1]['t']:.0f}s)", flush=True)

    cov_w = torch.stack(cov_w); cov_bc = torch.stack(cov_bc)
    ghat_all = torch.stack(ghat_all); se_all = torch.stack(se_all)
    ratio = se_all.mean(0) / ghat_all.std(0, unbiased=True).clamp_min(1e-12)

    def _s(c):
        return dict(overall=float(c.mean()), diag=float(c[:, :p].mean()), offdiag=float(c[:, p:].mean()))
    return dict(nominal=level, wald=_s(cov_w), bias_corrected=_s(cov_bc),
                se_ratio=dict(overall=float(ratio.median()), diag=float(ratio[:p].median()),
                              offdiag=float(ratio[p:].median())),
                records=recs)


def coverage_experiment(p, q, n, *, D=100, H=200, wz_scale=0.5, l2_coef=0.001,
                        device=None, level=0.95, truth_seed=0, verbose=True):
    """For each of D datasets ~ f_{theta_true}: fit theta_hat, bootstrap SE of the Gram
    functionals, form Wald CIs centered at the point estimate, and check coverage of the
    TRUE Gram values. Returns a dict with pooled coverage and per-dataset records."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    z_crit = float(torch.distributions.Normal(0, 1).icdf(torch.tensor(0.5 + level / 2)))

    # FIXED truth; the D datasets are fresh draws (z + noise) from this same model.
    truth = make_truth(p, q, wz_scale=wz_scale, seed=truth_seed)
    W_true = truth.wz.detach().cpu().clone()
    idx = select_gram_idx(p, q, seed=truth_seed)
    g_true = gram_entries(W_true, idx)                       # (m,)

    alpha = 1.0 - level
    cov_wald, cov_bc, cov_pct, widths, recs = [], [], [], [], []
    ghat_all, se_all = [], []   # for the bootstrap-SE vs empirical-SD calibration check
    for d in range(D):
        t0 = time.time()
        Y = sample_data(truth, n, seed=1000 + d)
        ft = fit_point(Y, q, l2=l2_coef / n, device=device, seed=1000 + d, wz_scale=wz_scale)
        W_hat = ft.model.wz.detach().cpu()
        g_hat = gram_entries(W_hat, idx)                     # (m,)  plug-in point estimate
        W_boot = param_bootstrap(ft, H, device=device, seed=7000 + d)
        g_boot = gram_entries(W_boot, idx)                   # (H, m)
        se = g_boot.std(0, unbiased=True)                    # (m,)  bootstrap SE

        # (1) plain Wald, centered at the plug-in estimate
        lo, hi = g_hat - z_crit * se, g_hat + z_crit * se
        cov_wald.append(((g_true >= lo) & (g_true <= hi)).float())

        # (2) bias-corrected Wald: subtract the bootstrap bias estimate
        #     bias_hat = E*[g(W*)] - g_hat ;  g_bc = g_hat - bias_hat = 2 g_hat - mean(g*)
        g_bc = 2.0 * g_hat - g_boot.mean(0)
        lo_bc, hi_bc = g_bc - z_crit * se, g_bc + z_crit * se
        cov_bc.append(((g_true >= lo_bc) & (g_true <= hi_bc)).float())

        # (3) basic/pivotal bootstrap interval (bias- and skew-aware, no normality):
        #     [2 g_hat - q_{1-a/2},  2 g_hat - q_{a/2}]
        qlo = torch.quantile(g_boot, alpha / 2, dim=0)
        qhi = torch.quantile(g_boot, 1 - alpha / 2, dim=0)
        lo_b, hi_b = 2.0 * g_hat - qhi, 2.0 * g_hat - qlo
        cov_pct.append(((g_true >= lo_b) & (g_true <= hi_b)).float())

        widths.append(hi - lo)
        ghat_all.append(g_hat); se_all.append(se)
        procW = procrustes_error(W_true, W_hat)
        recs.append(dict(d=d, procW=float(procW),
                         cov_wald=float(cov_wald[-1].mean()), cov_bc=float(cov_bc[-1].mean()),
                         cov_basic=float(cov_pct[-1].mean()),
                         med_se=float(se.median()), t=time.time() - t0))
        if verbose:
            print(f"[{d+1:>3}/{D}] procW={procW:.3f}  wald={cov_wald[-1].mean():.3f}  "
                  f"bc={cov_bc[-1].mean():.3f}  basic={cov_pct[-1].mean():.3f}  "
                  f"({recs[-1]['t']:.1f}s)", flush=True)

    cov_wald = torch.stack(cov_wald); cov_bc = torch.stack(cov_bc); cov_pct = torch.stack(cov_pct)

    def _summ(c):
        return dict(overall=float(c.mean()), diag=float(c[:, :p].mean()),
                    offdiag=float(c[:, p:].mean()))

    # SE calibration: mean bootstrap SE vs empirical across-dataset SD of the estimator.
    # ratio ~ 1 = bootstrap SE is right; < 1 = bootstrap underestimates the true variance.
    ghat_all = torch.stack(ghat_all)                     # (D, m)
    se_all = torch.stack(se_all)                         # (D, m)
    emp_sd = ghat_all.std(0, unbiased=True)              # (m,) empirical sampling SD
    mean_se = se_all.mean(0)                             # (m,) mean bootstrap SE
    ratio = mean_se / emp_sd.clamp_min(1e-12)            # (m,)
    se_cal = dict(diag=float(ratio[:p].median()), offdiag=float(ratio[p:].median()),
                  overall=float(ratio.median()))

    out = dict(
        nominal=level,
        wald=_summ(cov_wald),               # plain, centered at plug-in
        bias_corrected=_summ(cov_bc),       # bootstrap bias-corrected Wald
        basic=_summ(cov_pct),               # basic/pivotal bootstrap interval
        se_calibration=se_cal,              # boot-SE / empirical-SD (median); 1.0 = calibrated
        mean_width=float(torch.stack(widths).mean()),
        records=recs, idx=idx, g_true=g_true,
        p=p, q=q, n=n, D=D, H=H,
    )
    return out
