"""
simulation_7 — GLLVM scaling to very high response dimension via ZQE.

The estimator (``ZQEAutoFitter`` + parameter-free **Poisson-MAP** encoder) is run on a
**sparse** Poisson GLLVM and swept across the response dimension ``p`` at fixed ``q=10``,
``n=500``.  No competitor: R ``gllvm`` needs an O(p^2 q^2) Hessian and VA/Laplace an
intractable q-dimensional integral, so at ``q=10`` neither can even start in this regime
(and at p in the thousands–tens-of-thousands they are flatly infeasible).

Setup mirrors ``playground/poisson.ipynb`` exactly (only ``p`` is swept):

    q (latent)      = 10
    n (obs)         = 500
    wz_scale        = 0.5
    l2              = L2_COEF / n   (= 0.001/n, the tiny consistency-preserving ridge)
    lower_tri       = True          (pins the rotation gauge)
    encoder         = Poisson-MAP (MapEncoderPoissonNewton) — Poisson MAP ONLY
                      (Gaussian-log1p MAP explodes / recovers badly at large q)
    decoder T(y)    = log1p
    fitter knobs    = steps_per_round=150, max_rounds=2, tol=1e-3,
                      warmup_lr=refine_lr=0.5, ema_decay=0.95
                      (max_rounds=2: convergence is immediate after warm-up — see
                       plot_params; further refine restarts are wasted compute)

True model is **sparse**: ``make_sparse(..., active_latent=q,
responses_per_latent=p//2)`` — each latent loads on half the responses (overlapping),
all q latents active.  This keeps Poisson rates bounded at large p/q and matches the
sparse setting of poisson.ipynb (unlike simulation_1, which is dense).

The story: per-step cost is O(p) FLOPs but ~flat wall-clock on GPU (the p axis is
embarrassingly parallel); relative loading error (procW) *falls then plateaus* — once the
encoder pins z (~p>=1000 at q=10) it hits the n-limited oracle floor and stays flat (more
loadings AND more z-information, the sqrt(p) factors cancel in the relative metric).
Crucially it does **not** explode at large p.

Results persist as one tiny CSV per (p, rep) under ``results/`` (scalar summaries only —
the procW / time story needs no per-element W), so 20-rep runs are resumable.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from gllvm.gllvm_module import GLLVM
from gllvm.glms import PoissonGLM
from gllvm.encoder import MapEncoderPoissonNewton
from gllvm.autofit import ZQEAutoFitter, procrustes_error
from gllvm.simulations import make_sparse, simulate

# ----------------------------------------------------------------------------
# Sweep config (mirror poisson.ipynb; only p is swept)
# ----------------------------------------------------------------------------
Q          = 10
N          = 500
WZ_SCALE   = 0.5
L2_COEF    = 0.001     # per-fit ridge = L2_COEF / n
LOWER_TRI  = True
P_GRID     = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]

# max_rounds=2: plot_params shows loadings lock onto truth during warm-up + the first
# refine round; further restarts are wasted compute (warm-up + 2 refine is enough).
ZQE_KW = dict(steps_per_round=150, max_rounds=2, tol=0.001,
              warmup_lr=0.5, refine_lr=0.5, ema_decay=0.95, verbose=False,
              store_wz_trace=False)

HERE        = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")


# ----------------------------------------------------------------------------
# Data / fit (one (p, rep))
# ----------------------------------------------------------------------------
def make_data(p: int, q: int, n: int, *, wz_scale: float = WZ_SCALE,
              seed: int = 0, lower_tri: bool = LOWER_TRI):
    """Sparse Poisson GLLVM + dataset, generated on CPU (reproducible; avoids the
    CUDA heavy-tail draw that can fall into a spurious Poisson-MAP root).
    Returns ``(Y_cpu, W_true)`` with ``W_true`` a CPU tensor used only for scoring."""
    torch.manual_seed(seed)
    g_true = make_sparse(n_latent=q, poisson=p, active_latent=q,
                         wz_scale=wz_scale, responses_per_latent=p // 2,
                         lower_tri=lower_tri)
    Y, _ = simulate(g_true, n_samples=n, device="cpu")
    return Y, g_true.wz.detach().cpu().clone()


def fit(Y, q: int, *, l2: float, device: str, seed: int = 0, W_true=None,
        wz_scale: float = WZ_SCALE, lower_tri: bool = LOWER_TRI, **zqe_kw):
    """Fresh sparse-init Poisson GLLVM fit by ZQE with the **Poisson-MAP** encoder, using
    all p responses per step.  Returns ``(fitter, fit_time_seconds, procW_or_None)``."""
    p = Y.shape[1]
    torch.manual_seed(seed)
    g = GLLVM(latent_dim=q, output_dim=p, bias=True, lower_tri=lower_tri).to(device)
    g.add_glm(PoissonGLM, idx=list(range(p)), params={"T": torch.log1p}, name="P")
    with torch.no_grad():
        nn.init.normal_(g.wz, std=wz_scale)
        nn.init.zeros_(g.bias)
    kw = {**ZQE_KW, **zqe_kw}
    t0 = time.perf_counter()
    ft = ZQEAutoFitter(
        g, encoder_factory=lambda g: MapEncoderPoissonNewton(g, lam=1.0, max_iter=30),
        device=device, seed=seed, l2=l2, **kw,
    ).fit(Y.to(device))
    dt = time.perf_counter() - t0
    pw = None if W_true is None else procrustes_error(W_true, ft.model.wz)
    return ft, dt, pw


# ----------------------------------------------------------------------------
# Persistence / sweep (resumable, one CSV per (p, rep))
# ----------------------------------------------------------------------------
def result_path(q: int, p: int, n: int, rep: int) -> str:
    return os.path.join(RESULTS_DIR, f"q{q}_p{p}_n{n}_rep{rep:03d}.csv")


def _save_atomic(df: pd.DataFrame, path: str) -> None:
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def run_one(p: int, rep: int, *, q: int = Q, n: int = N, device: str,
            wz_scale: float = WZ_SCALE, l2_coef: float = L2_COEF,
            lower_tri: bool = LOWER_TRI, **zqe_kw) -> dict:
    """Run a single (p, rep): seed = rep, l2 = l2_coef/n.  Returns a one-row record."""
    seed = rep
    Y, W_true = make_data(p, q, n, wz_scale=wz_scale, seed=seed, lower_tri=lower_tri)
    mc = float(Y.float().mean())
    ft, dt, pw = fit(Y, q, l2=l2_coef / n, device=device, seed=seed, W_true=W_true,
                     wz_scale=wz_scale, lower_tri=lower_tri, **zqe_kw)
    return dict(q=q, p=p, n=n, rep=rep, seed=seed, procW=float(pw), fit_s=dt,
                converged=bool(ft.converged_), n_rounds=int(ft.n_rounds_used_),
                w_true_norm=float(W_true.norm()),
                w_hat_norm=float(ft.model.wz.detach().norm()), mean_count=mc)


def run_sweep(reps: int, *, p_grid=P_GRID, q: int = Q, n: int = N,
              device: str | None = None, overwrite: bool = False,
              verbose: bool = True, **zqe_kw) -> None:
    """Sweep p × reps; skips (p, rep) whose CSV already exists (resumable)."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    todo = [(p, rep) for p in p_grid for rep in range(reps)
            if overwrite or not os.path.exists(result_path(q, p, n, rep))]
    if verbose:
        print(f"device={device}  grid={len(p_grid)} p × {reps} reps; "
              f"{len(p_grid)*reps - len(todo)} done, {len(todo)} to run.", flush=True)
    for k, (p, rep) in enumerate(todo, 1):
        t0 = time.time()
        row = run_one(p, rep, q=q, n=n, device=device, **zqe_kw)
        _save_atomic(pd.DataFrame([row]), result_path(q, p, n, rep))
        if verbose:
            print(f"[{k:>3}/{len(todo)}] p={p:>6d} rep{rep:03d}  procW={row['procW']:.3f}  "
                  f"fit={row['fit_s']:6.1f}s  conv={row['converged']}  "
                  f"|Ŵ|/|W|={row['w_hat_norm']/row['w_true_norm']:.2f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)


def load_results(q: int = Q, n: int = N) -> pd.DataFrame:
    import glob
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, f"q{q}_p*_n{n}_rep*.csv")))
    if not files:
        raise FileNotFoundError(f"no result CSVs in {RESULTS_DIR!r}; run the sweep first.")
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
