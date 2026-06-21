"""
sweep.py — driver for *simulation_1* (Poisson loadings recovery, "gllvm's turf").

This module is **specific to this experiment** and deliberately lives outside the
``gllvm`` source package.  It wraps the per-setting fit so the notebooks stay
tidy, and owns the on-disk result layout so the sweep is **reproducible** and
**resumable** (growing H from 20 → 100 only fits the new reps).

Design at a glance
------------------
* One fully-dense Poisson GLLVM per (setting, rep); ``seed == rep`` drives both
  the *true model* and the simulated data, so every rep is independently
  reproducible from its seed alone.
* Two estimators, fit to the *same* data:
    - **ZQE**  : ``ZQEAutoFitter`` with the Gaussian-MAP encoder and ``T=log1p``
                 on both decoder and encoder — exactly the spec in
                 ``simulations/poisson.ipynb``.
    - **gllvm**: R's ``gllvm()`` via :class:`gllvm.r_gllvm.RGllvm` (method "VA").
* For each rep we store, per method **and** for the true model, every parameter
  flattened: the loadings (Procrustes-rotated into the true gauge) and the
  intercepts.  Plus the wall-clock fit time and (for ZQE) convergence.

Result layout
-------------
``results/q{q}_p{p}_n{n}_rep{rep:03d}.csv`` — **one CSV per (setting, rep)**.
A file's existence means that rep is done; that is the whole resumability story
(no shared file to append to, no partial-write races).  Column meanings are in
``results/DATA_DICTIONARY.md``.  ``load_results()`` globs and concatenates them.

Each CSV is in **long / tidy** form — one row per scalar parameter:

    q, p, n, rep, seed, method, failed, time_sec, converged, procrustes, param, i, j, value

so ``value`` is one entry of either the loading matrix W (``param='W'``, row ``i``
∈[0,p), col ``j``∈[0,q)) or the intercept vector b (``param='b'``, ``i``∈[0,p),
``j=-1``).  ``failed``/``time_sec``/``converged``/``procrustes`` are per-fit scalars
repeated on every row of that (rep, method) block; they are ``NaN`` for
``method='true'``.

Robustness: a fit that raises (R error, NaN blow-up, …) does **not** abort the
sweep — the rep is still written, with that method flagged ``failed=1.0`` and all
of its ``value``/``time_sec``/``converged``/``procrustes`` set to ``NaN``.  The
true block and the other method are unaffected.
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
from gllvm.encoder import MapEncoderGaussianLog1p
from gllvm.autofit import ZQEAutoFitter, orthogonal_align, procrustes_error
from gllvm.r_gllvm import RGllvm

# ----------------------------------------------------------------------------
# Sweep grid (start in "gllvm's turf"; widen later by editing these lists)
# ----------------------------------------------------------------------------
Q = 2                       # latent dimension (fixed)
P_GRID = [10, 20, 50, 100]  # responses
N_GRID = [20, 100, 500]     # observations
WZ_SCALE = 1.0              # loading scale (matches simulations/poisson.ipynb)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

# ZQE fitter knobs — identical to simulations/poisson.ipynb.
ZQE_KW = dict(steps_per_round=150, max_rounds=10, tol=0.001,
              refine_lr=0.5, warmup_lr=0.5, ema_decay=0.95, verbose=False)


# ----------------------------------------------------------------------------
# Encoder (experiment-local: sim wrappers stay out of the source package)
# ----------------------------------------------------------------------------
class QuantileMapEncoder(nn.Module):
    """Gaussian-MAP (log1p) encoder, then marginally project each latent
    dimension to the prior ``N(0,1)`` via a rank -> ``Phi^{-1}`` (PIT) transform.

    Same MAP as the ``zqe`` arm (:class:`MapEncoderGaussianLog1p`), so the *only*
    difference vs that arm is the projection — which recalibrates the badly
    over-shrunk MAP marginal back to the prior (see
    ``playground/poisson_encoder_quantile.ipynb``).  The projection depends only
    on the *ranks* of the MAP scores, hence is non-differentiable — harmless,
    since ``ZQEAutoFitter`` calls ``encoder.sample`` under ``torch.no_grad``
    (the score-function identity: the encoder is a detached surrogate).
    """

    def __init__(self, gllvm, sigma2: float = 1.0):
        super().__init__()
        self.gllvm = gllvm
        self.base = MapEncoderGaussianLog1p(gllvm, sigma2=sigma2)

    def _project(self, z):                       # per-dim rank -> N(0,1)
        n = z.shape[0]
        ranks = z.argsort(0).argsort(0).double()
        return torch.special.ndtri((ranks + 0.5) / n).to(z.dtype)

    def sample(self, y):
        z = self._project(self.base(y))
        return z, z, torch.full_like(z, float("-inf"))

    def loss(self, y, gllvm=None, **kwargs):
        dummy = next(self.gllvm.parameters())
        return torch.zeros(1, device=dummy.device, requires_grad=True), 0.0


# ----------------------------------------------------------------------------
# Result-file bookkeeping
# ----------------------------------------------------------------------------
def result_path(q: int, p: int, n: int, rep: int) -> str:
    return os.path.join(RESULTS_DIR, f"q{q}_p{p}_n{n}_rep{rep:03d}.csv")


def is_done(q: int, p: int, n: int, rep: int) -> bool:
    """A rep is done iff its CSV already exists."""
    return os.path.exists(result_path(q, p, n, rep))


# ----------------------------------------------------------------------------
# Model / data
# ----------------------------------------------------------------------------
def fresh_decoder(q: int, p: int, device: str) -> GLLVM:
    """Random dense Poisson decoder with the ZQE statistic ``T(y)=log1p``.

    Mirrors ``fresh_decoder`` in ``simulations/poisson.ipynb``: random init, no
    knowledge of the truth; ``T`` only sets the estimating function (sampling /
    log_prob are unchanged).
    """
    g = GLLVM(latent_dim=q, output_dim=p, bias=True).to(device)
    g.add_glm(PoissonGLM, idx=list(range(p)), params={"T": torch.log1p}, name="P")
    with torch.no_grad():
        nn.init.normal_(g.wz, std=WZ_SCALE)
        nn.init.zeros_(g.bias)
    return g


def simulate_setting(q: int, p: int, n: int, seed: int):
    """Build one true dense Poisson GLLVM and draw a dataset, on CPU.

    Generated on CPU so the dataset is identical on any machine (the torch RNG
    stream differs between CPU and CUDA).  Returns ``(Y_cpu, W_true, b_true)``
    with ``W_true`` (p,q) and ``b_true`` (p,) as float64 numpy arrays.
    """
    from gllvm.simulations import make_mixed, simulate
    torch.manual_seed(seed)
    g_true = make_mixed(n_latent=q, poisson=p, wz_scale=WZ_SCALE)
    Y, _ = simulate(g_true, n_samples=n, device="cpu")
    W_true = g_true.wz.detach().to(torch.float64).numpy()
    b_true = g_true.bias.detach().to(torch.float64).numpy()
    return Y, W_true, b_true


# ----------------------------------------------------------------------------
# Estimators (each returns aligned loadings, intercepts, time, converged)
# ----------------------------------------------------------------------------
def _align(W_true: np.ndarray, W_est) -> np.ndarray:
    """Rotate ``W_est`` into the true gauge: ``W_est @ R*`` (float64 numpy)."""
    Wt = torch.as_tensor(np.asarray(W_true), dtype=torch.float64)
    We = (W_est.detach().to("cpu", torch.float64) if isinstance(W_est, torch.Tensor)
          else torch.as_tensor(np.asarray(W_est), dtype=torch.float64))
    R = orthogonal_align(Wt, We)
    return (We @ R).numpy()


def fit_zqe(Y: torch.Tensor, q: int, p: int, seed: int, device: str,
            W_true: np.ndarray):
    """ZQE fit (Gaussian-MAP encoder, T=log1p) on data ``Y``.

    Returns ``(W_aligned, b, time_sec, converged)``.
    """
    torch.manual_seed(seed)                  # reproducible decoder init
    g = fresh_decoder(q, p, device)
    t0 = time.time()
    ft = ZQEAutoFitter(g, encoder_factory=lambda g: MapEncoderGaussianLog1p(g),
                       device=device, seed=seed, **ZQE_KW).fit(Y.to(device))
    dt = time.time() - t0
    W = _align(W_true, ft.model.wz)
    b = ft.model.bias.detach().to("cpu", torch.float64).numpy()
    return W, b, dt, bool(ft.converged_)


def fit_zqe_q(Y: torch.Tensor, q: int, p: int, seed: int, device: str,
              W_true: np.ndarray):
    """ZQE fit with the quantile-projected Gaussian-MAP encoder (T=log1p).

    Identical to :func:`fit_zqe` except the encoder is
    :class:`QuantileMapEncoder` (MAP + per-dim N(0,1) projection).
    """
    torch.manual_seed(seed)
    g = fresh_decoder(q, p, device)
    t0 = time.perf_counter()                  # monotonic: immune to WSL2 clock skew
    ft = ZQEAutoFitter(g, encoder_factory=lambda g: QuantileMapEncoder(g),
                       device=device, seed=seed, **ZQE_KW).fit(Y.to(device))
    dt = time.perf_counter() - t0
    W = _align(W_true, ft.model.wz)
    b = ft.model.bias.detach().to("cpu", torch.float64).numpy()
    return W, b, dt, bool(ft.converged_)


def fit_gllvm(Y: torch.Tensor, q: int, seed: int, rgllvm: RGllvm,
              W_true: np.ndarray):
    """R ``gllvm()`` fit on data ``Y``.

    Returns ``(W_aligned, b, time_sec, converged=nan)`` — gllvm has no
    ZQE-style convergence flag, so ``converged`` is left ``NaN``.
    """
    t0 = time.time()
    rf = rgllvm.fit(Y.cpu().numpy(), num_lv=q, seed=seed)
    dt = time.time() - t0
    W = _align(W_true, rf.loadings)
    b = (np.asarray(rf.intercepts, dtype=float)
         if rf.intercepts is not None else np.full(W.shape[0], np.nan))
    return W, b, dt, float("nan")


# ----------------------------------------------------------------------------
# Record assembly (long / tidy: one row per scalar parameter)
# ----------------------------------------------------------------------------
def _rows(q, p, n, rep, seed, method, W, b, time_sec, converged, procr,
          failed=0.0):
    """Flatten one fit's (W, b) into long-format row dicts.

    ``W`` / ``b`` may be ``None`` (a failed fit): every ``value`` is then ``NaN``
    but the rows still exist, so the parameter grid is complete for every method.
    """
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
                rgllvm: RGllvm, methods=("zqe", "zqe_q", "gllvm"),
                overwrite: bool = False):
    """Fit one (setting, rep), **merging** with any existing result file.

    Only methods *missing* from the existing file are fitted (unless
    ``overwrite``), so a new arm (e.g. ``zqe_q``) is appended to already-finished
    reps **without recomputing** ``zqe``/``gllvm``.  The ``true`` block is
    computed once and kept thereafter.

    Returns the full (merged) long-format DataFrame, or ``None`` when the file
    already contains everything requested (nothing to do).
    """
    path = result_path(q, p, n, seed)
    existing = pd.read_csv(path) if os.path.exists(path) else None
    present = set(existing.method.unique()) if existing is not None else set()
    to_run = [m for m in methods if overwrite or m not in present]
    if not to_run and "true" in present:
        return None                                  # nothing new

    Y, W_true, b_true = simulate_setting(q, p, n, seed)  # cheap, seed-reproducible

    rows = []
    if "true" not in present:
        rows += _rows(q, p, n, seed, seed, "true", W_true, b_true,
                      np.nan, np.nan, np.nan, failed=np.nan)

    fitters = {"zqe":   lambda: fit_zqe(Y, q, p, seed, device, W_true),
               "zqe_q": lambda: fit_zqe_q(Y, q, p, seed, device, W_true),
               "gllvm": lambda: fit_gllvm(Y, q, seed, rgllvm, W_true)}
    for m in to_run:
        try:
            W, b, dt, conv = fitters[m]()
            rows += _rows(q, p, n, seed, seed, m, W, b, dt, conv,
                          procrustes_error(W_true, W), failed=0.0)
        except Exception as e:                       # flag, don't abort the sweep
            print(f"    !! {m} FAILED at q{q} p{p} n{n} rep{seed}: "
                  f"{type(e).__name__}: {e}")
            rows += _rows(q, p, n, seed, seed, m, None, None,
                          np.nan, np.nan, np.nan, failed=1.0)

    new = pd.DataFrame(rows)
    if existing is not None:                          # keep all non-rerun blocks
        keep = existing[~existing.method.isin(to_run)]
        return pd.concat([keep, new], ignore_index=True)
    return new


def _save_atomic(df: pd.DataFrame, path: str) -> None:
    """Write CSV via a temp file + rename so a killed run leaves no half file."""
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------
# Sweep driver
# ----------------------------------------------------------------------------
def _present_methods(q: int, p: int, n: int, rep: int) -> set:
    """Methods already stored in a rep's CSV (empty set if the file is absent)."""
    path = result_path(q, p, n, rep)
    if not os.path.exists(path):
        return set()
    return set(pd.read_csv(path, usecols=["method"]).method.unique())


def run_sweep(reps: int, *, q: int = Q, p_grid=P_GRID, n_grid=N_GRID,
              device: Optional[str] = None, methods=("zqe", "zqe_q", "gllvm"),
              overwrite: bool = False, rgllvm: Optional[RGllvm] = None,
              verbose: bool = True) -> None:
    """Run the full grid for ``rep`` in ``range(reps)``, skipping finished work.

    Resumable **and method-aware**: a rep is run only if some requested method
    (or ``true``) is missing from its CSV.  Bumping ``reps`` 20 → 100 fits the new
    reps; adding a method (e.g. ``zqe_q``) appends *only that method* to existing
    reps, leaving ``zqe``/``gllvm`` untouched.  Run a single new arm cheaply with
    ``methods=("zqe_q",)`` (no R needed — ``RGllvm`` is built only if ``gllvm`` is
    requested).
    """
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
              f"= {total};  {total - len(todo)} complete, {len(todo)} to (re)run "
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
                (f"{m}:FAILED" if fl == 1.0 else f"{m}:P={pr:.3f}/{t:.0f}s")
                for m, fl, pr, t in zip(sub.method, sub.failed,
                                        sub.procrustes, sub.time_sec))
            print(f"[{k:>4}/{len(todo)}] q{q} p{p:<3} n{n:<3} rep{rep:03d}  "
                  f"{tag}  ({time.time()-t0:.0f}s)")


# ----------------------------------------------------------------------------
# Loading results back
# ----------------------------------------------------------------------------
def load_results(q: int = Q, p_grid=P_GRID, n_grid=N_GRID) -> pd.DataFrame:
    """Concatenate every per-rep CSV under ``results/`` into one DataFrame."""
    import glob
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, f"q{q}_p*_n*_rep*.csv")))
    if not files:
        raise FileNotFoundError(f"no result CSVs in {RESULTS_DIR!r}; run the sweep first.")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    return df
