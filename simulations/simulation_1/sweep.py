"""
sweep.py — driver for *simulation_1* (Poisson loadings recovery).

Three arms on a dense Poisson GLLVM (lower-triangular true W, q=2), mirroring the
setup in ``playground/poisson.ipynb``:

    zqe_pmap   ZQE (ZQEAutoFitter), T=log1p, Poisson-MAP encoder, l2=L2_COEF/n
    zqe_gmap   ZQE (ZQEAutoFitter), T=log1p, Gaussian-log1p-MAP encoder, l2=L2_COEF/n
    gllvm      R gllvm (VA, Poisson)

Both ZQE arms share the identical ZQEAutoFitter recipe and the same tiny ridge
``l2 = L2_COEF/n`` — only the parameter-free encoder differs.  The ridge is a
consistency-preserving stabilizer: scaled as c/n its bias is O(1/n) ≪ the O(1/√n)
standard error, so it removes the near-unidentifiability divergence at essentially
no bias.

Grid is p∈{10,20,50} so gllvm stays in its identifiable regime and the comparison
is fair.  L2_COEF=0.001 (penalty = 0.001/n, as in poisson.ipynb).  lower_tri=True
pins the rotation gauge (matches R gllvm's default identifiability constraint).
"""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from gllvm.gllvm_module import GLLVM
from gllvm.glms import PoissonGLM
from gllvm.encoder import MapEncoderGaussianLog1p, MapEncoderPoissonNewton
from gllvm.autofit import ZQEAutoFitter, orthogonal_align, procrustes_error
from gllvm.r_gllvm import RGllvm

# ----------------------------------------------------------------------------
# Sweep grid
# ----------------------------------------------------------------------------
Q          = 2
P_GRID     = [10, 20, 50]
N_GRID     = [20, 100, 500]
WZ_SCALE   = 0.5
L2_COEF    = 0.001  # per-fit penalty = L2_COEF / n  (tiny stabilizer, as in poisson.ipynb)
LOWER_TRI  = True   # pins rotation gauge; matches R gllvm's default constraint

HERE        = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

ZQE_KW = dict(steps_per_round=150, max_rounds=10, tol=0.001,
              refine_lr=0.5, warmup_lr=0.5, ema_decay=0.95, verbose=False)

DEFAULT_METHODS = ("zqe_pmap", "zqe_gmap", "gllvm")


# encoder factories (mirror poisson.ipynb)
def _enc_pmap(g):
    return MapEncoderPoissonNewton(g, lam=1.0, max_iter=30)


def _enc_gmap(g):
    return MapEncoderGaussianLog1p(g)


# module-level T function (lambda-free → deepcopy-safe)
def _T_log1p(y):
    return torch.log1p(y.float())


# ----------------------------------------------------------------------------
# Result-file bookkeeping
# ----------------------------------------------------------------------------
def result_path(q: int, p: int, n: int, rep: int) -> str:
    return os.path.join(RESULTS_DIR, f"q{q}_p{p}_n{n}_rep{rep:03d}.csv")


def _present_methods(q: int, p: int, n: int, rep: int) -> set:
    path = result_path(q, p, n, rep)
    if not os.path.exists(path):
        return set()
    return set(pd.read_csv(path, usecols=["method"]).method.unique())


# ----------------------------------------------------------------------------
# Model / data
# ----------------------------------------------------------------------------
def fresh_decoder(q: int, p: int, device: str, T_fn=_T_log1p) -> GLLVM:
    g = GLLVM(latent_dim=q, output_dim=p, bias=True, lower_tri=LOWER_TRI).to(device)
    g.add_glm(PoissonGLM, idx=list(range(p)), params={"T": T_fn}, name="P")
    with torch.no_grad():
        nn.init.normal_(g.wz, std=WZ_SCALE)
        nn.init.zeros_(g.bias)
    return g


def simulate_setting(q: int, p: int, n: int, seed: int):
    """Dense lower-triangular Poisson GLLVM + dataset on CPU."""
    from gllvm.simulations import make_sparse, simulate
    torch.manual_seed(seed)
    g_true = make_sparse(n_latent=q, poisson=p, active_latent=q,
                         wz_scale=WZ_SCALE, responses_per_latent=p,
                         lower_tri=LOWER_TRI)
    Y, _ = simulate(g_true, n_samples=n, device="cpu")
    W_true = g_true.wz.detach().to(torch.float64).numpy()
    b_true = g_true.bias.detach().to(torch.float64).numpy()
    return Y, W_true, b_true


# ----------------------------------------------------------------------------
# Estimators
# ----------------------------------------------------------------------------
def _align(W_true: np.ndarray, W_est) -> np.ndarray:
    Wt = torch.as_tensor(np.asarray(W_true), dtype=torch.float64)
    We = (W_est.detach().to("cpu", torch.float64) if isinstance(W_est, torch.Tensor)
          else torch.as_tensor(np.asarray(W_est), dtype=torch.float64))
    return (We @ orthogonal_align(Wt, We)).numpy()


def _fit_zqe(Y, q, p, seed, device, W_true, enc_factory, l2):
    torch.manual_seed(seed)
    g = fresh_decoder(q, p, device, T_fn=_T_log1p)
    t0 = time.time()
    ft = ZQEAutoFitter(g, encoder_factory=enc_factory,
                       device=device, seed=seed, l2=l2,
                       **ZQE_KW).fit(Y.to(device))
    dt = time.time() - t0
    W = _align(W_true, ft.model.wz)
    b = ft.model.bias.detach().to("cpu", torch.float64).numpy()
    return W, b, dt, bool(ft.converged_)


def fit_zqe_pmap(Y, q, p, seed, device, W_true):
    return _fit_zqe(Y, q, p, seed, device, W_true, _enc_pmap, L2_COEF / Y.shape[0])

def fit_zqe_gmap(Y, q, p, seed, device, W_true):
    return _fit_zqe(Y, q, p, seed, device, W_true, _enc_gmap, L2_COEF / Y.shape[0])


def fit_gllvm(Y, q, seed, rgllvm, W_true):
    t0 = time.time()
    rf = rgllvm.fit(Y.cpu().numpy(), num_lv=q, seed=seed)
    dt = time.time() - t0
    W = _align(W_true, rf.loadings)
    b = (np.asarray(rf.intercepts, dtype=float)
         if rf.intercepts is not None else np.full(W.shape[0], np.nan))
    return W, b, dt, float("nan")


# ----------------------------------------------------------------------------
# Record assembly
# ----------------------------------------------------------------------------
def _rows(q, p, n, rep, seed, method, W, b, time_sec, converged, procr, failed=0.0):
    rows = []
    common = dict(q=q, p=p, n=n, rep=rep, seed=seed, method=method,
                  failed=failed, time_sec=time_sec, converged=converged,
                  procrustes=procr)
    W = None if W is None else np.asarray(W)
    for i in range(p):
        for j in range(q):
            v = np.nan if W is None else float(W[i, j])
            rows.append({**common, "param": "W", "i": i, "j": j, "value": v})
    b = None if b is None else np.asarray(b)
    for i in range(p):
        v = np.nan if b is None else float(b[i])
        rows.append({**common, "param": "b", "i": i, "j": -1, "value": v})
    return rows


def run_setting(q: int, p: int, n: int, seed: int, device: str,
                rgllvm: RGllvm, methods=DEFAULT_METHODS,
                overwrite: bool = False):
    path = result_path(q, p, n, seed)
    existing = pd.read_csv(path) if os.path.exists(path) else None
    present = set(existing.method.unique()) if existing is not None else set()
    to_run = [m for m in methods if overwrite or m not in present]
    if not to_run and "true" in present:
        return None

    Y, W_true, b_true = simulate_setting(q, p, n, seed)

    rows = []
    if "true" not in present:
        rows += _rows(q, p, n, seed, seed, "true", W_true, b_true,
                      np.nan, np.nan, np.nan, failed=np.nan)

    fitters = {
        "zqe_pmap": lambda: fit_zqe_pmap(Y, q, p, seed, device, W_true),
        "zqe_gmap": lambda: fit_zqe_gmap(Y, q, p, seed, device, W_true),
        "gllvm":    lambda: fit_gllvm(Y, q, seed, rgllvm, W_true),
    }
    for m in to_run:
        try:
            W, b, dt, conv = fitters[m]()
            rows += _rows(q, p, n, seed, seed, m, W, b, dt, conv,
                          procrustes_error(W_true, W), failed=0.0)
        except Exception as e:
            print(f"    !! {m} FAILED at q{q} p{p} n{n} rep{seed}: "
                  f"{type(e).__name__}: {e}")
            rows += _rows(q, p, n, seed, seed, m, None, None,
                          np.nan, np.nan, np.nan, failed=1.0)

    new = pd.DataFrame(rows)
    if existing is not None:
        keep = existing[~existing.method.isin(to_run)]
        return pd.concat([keep, new], ignore_index=True)
    return new


def _save_atomic(df: pd.DataFrame, path: str) -> None:
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------
# Sweep driver
# ----------------------------------------------------------------------------
def run_sweep(reps: int, *, q: int = Q, p_grid=P_GRID, n_grid=N_GRID,
              device: Optional[str] = None,
              methods=DEFAULT_METHODS,
              overwrite: bool = False, rgllvm: Optional[RGllvm] = None,
              verbose: bool = True) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if rgllvm is None and "gllvm" in methods:
        rgllvm = RGllvm(method="VA", family="poisson")
        if not rgllvm.available():
            raise RuntimeError(
                f"R Rscript not found at {rgllvm.rscript!r}; pass an RGllvm with "
                "rscript=.../workdir=..., or drop 'gllvm' from methods."
            )

    settings = [(p, n) for p in p_grid for n in n_grid]
    need = set(methods) | {"true"}
    todo = [(p, n, rep) for (p, n) in settings for rep in range(reps)
            if overwrite or not need.issubset(_present_methods(q, p, n, rep))]
    if verbose:
        total = len(settings) * reps
        print(f"device={device}  grid={len(settings)} settings × {reps} reps "
              f"= {total};  {total - len(todo)} complete, {len(todo)} to run "
              f"[methods={','.join(methods)}].")

    for k, (p, n, rep) in enumerate(todo, 1):
        t0 = time.time()
        df = run_setting(q, p, n, rep, device, rgllvm, methods=methods,
                         overwrite=overwrite)
        if df is None:
            continue
        _save_atomic(df, result_path(q, p, n, rep))
        if verbose:
            sub = df[df.method != "true"].drop_duplicates("method")
            tag = "  ".join(
                (f"{m}:FAIL" if fl == 1.0 else f"{m}:P={pr:.3f}/{t:.0f}s")
                for m, fl, pr, t in zip(sub.method, sub.failed,
                                        sub.procrustes, sub.time_sec))
            print(f"[{k:>4}/{len(todo)}] q{q} p{p:<3} n{n:<3} rep{rep:03d}  "
                  f"{tag}  ({time.time()-t0:.0f}s)")


def load_results(q: int = Q, p_grid=P_GRID, n_grid=N_GRID) -> pd.DataFrame:
    import glob
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, f"q{q}_p*_n*_rep*.csv")))
    if not files:
        raise FileNotFoundError(f"no result CSVs in {RESULTS_DIR!r}; run the sweep first.")
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
