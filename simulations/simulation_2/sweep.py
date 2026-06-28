"""
sweep.py — driver for *simulation_2* (robustness to outlier contamination).

Three branches, all fitting the same misspecified Poisson model to the same
contaminated data; they differ only in the **influence** of the response on the
estimating equation:

    gllvm      — R gllvm() (VA). Full Poisson likelihood → influence UNBOUNDED in y.
    zqe        — ZQE, T = log1p(y).            Logarithmic (down-weighted) influence.
    zqe_huber  — ZQE, T = min(log1p(y), c).    BOUNDED influence (Huberised log1p),
                 with c a robust cut in log space (median + 3·MAD of log1p(y)),
                 applied in both decoder and encoder.

The point: a bounded T gives a bounded influence function, so the estimator is
**flat against contamination until its breakdown point** — the outlier magnitude
M cannot move it at all, and the contaminated fraction eps can only move it once
eps is large enough to corrupt the robust scale c itself (the median/MAD break
near eps ≈ 0.5). The model expectation term m₂ supplies the Fisher-consistency
correction for the chosen T automatically (it is exactly E_θ[T·η]), so the
Huberised estimator stays consistent — this is the constructive instance of the
paper's robustness–efficiency trade-off.

Two dose-response sweeps share the clean point (eps=0, M=0):

    eps line (corrupted fraction):  eps ∈ {.02,.05,.10,.20,.30,.40,.50} at M = M_FIX
        → shows the breakdown of the robust arm as eps grows.
    mag line (outlier magnitude) :  M ∈ {10,100,1000,1e4,1e5}          at eps = EPS_FIX
        → shows the robust arm DEAD FLAT in M (bounded influence), while gllvm
          climbs unboundedly and plain log1p climbs logarithmically.

Same conventions as simulation 1: seed == rep drives model/data/contamination;
one CSV per (condition, rep); resumable; failures flagged (gllvm DNFs under heavy
contamination); loadings scored vs the clean truth after Procrustes rotation.
Columns: see ``results/DATA_DICTIONARY.md``.
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
# Fixed clean model + contamination grid
# ----------------------------------------------------------------------------
Q, P, N = 2, 50, 200
WZ_SCALE = 1.0

M_FIX = 1000           # gross outlier magnitude held fixed along the eps line
EPS_FIX = 0.05         # contamination fraction held fixed along the mag line
EPS_GRID = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]   # → breakdown of robust arm
M_GRID = [10, 100, 1000, 10_000, 100_000]               # → flatness of robust arm

HUBER_K = 3.0          # robust cut: c = median(log1p y) + HUBER_K · MAD(log1p y)
PEARSON_C = 2.0        # Huber tuning constant for the Pearson-residual weight (zqe_pearson)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

ZQE_KW = dict(steps_per_round=150, max_rounds=10, tol=0.001,
              refine_lr=0.5, warmup_lr=0.5, ema_decay=0.95, verbose=False)


def _conditions():
    """clean + eps-line + mag-line, de-duplicating the crossing (EPS_FIX, M_FIX)."""
    conds = [(0.0, 0)]
    conds += [(e, M_FIX) for e in EPS_GRID]
    conds += [(EPS_FIX, m) for m in M_GRID]
    seen, out = set(), []
    for e, m in conds:
        key = (round(e, 6), int(m))
        if key not in seen:
            seen.add(key); out.append((float(e), int(m)))
    return out

CONDITIONS = _conditions()


# ----------------------------------------------------------------------------
# Huberised-log1p statistic and its robust cut
# ----------------------------------------------------------------------------
def huber_cut(Y: torch.Tensor, k: float = HUBER_K) -> float:
    """Robust upper cut in log space: ``median + k·MAD`` of ``log1p(Y)``.

    Computed from the observed (contaminated) data; the median/MAD sit among the
    clean cells until the contaminated fraction approaches ~0.5, which is exactly
    where the robust arm breaks down.
    """
    t = torch.log1p(Y.float()).flatten()
    med = torch.median(t)
    mad = torch.median((t - med).abs()) * 1.4826
    return float(med + k * mad)


def make_T_huber(c: float):
    """T(y) = min(log1p(y), c) — Huberised (bounded) log1p."""
    return lambda y, c=c: torch.clamp(torch.log1p(y.float()), max=c)


# ----------------------------------------------------------------------------
# Encoders
# ----------------------------------------------------------------------------
class MapEncoderHuber(MapEncoderGaussianLog1p):
    """``MapEncoderGaussianLog1p`` on ``min(log1p(y), c)`` — bounded-influence E-step."""

    def __init__(self, gllvm, c, sigma2: float = 1.0):
        super().__init__(gllvm, sigma2)
        self.c = float(c)

    def forward(self, y):
        W = self.gllvm.wz
        b = (self.gllvm.bias if self.gllvm.bias is not None
             else torch.zeros(W.shape[0], device=W.device, dtype=W.dtype))
        t_y = torch.clamp(torch.log1p(y.float()), max=self.c)
        rhs = (t_y - b.unsqueeze(0)) @ W
        A = (self.sigma2 * torch.eye(W.shape[1], device=W.device, dtype=W.dtype)
             + W.T @ W)
        return torch.linalg.solve(A, rhs.T).T


class QuantileMapEncoder(MapEncoderGaussianLog1p):
    """Gaussian-MAP (log1p) then per-dim rank→N(0,1) projection — a *calibrated*
    latent (de-shrinks the MAP).  Used ONLY to form a robust μ for the Pearson
    residual: its scale must be right for ``|r| > c`` to mean what it should."""

    def forward(self, y):
        z = super().forward(y)
        n = z.shape[0]
        ranks = z.argsort(0).argsort(0).double()
        return torch.special.ndtri((ranks + 0.5) / n).to(z.dtype)


# ----------------------------------------------------------------------------
# Pearson-residual Huber weight (classic robust-GLM influence bound)
# ----------------------------------------------------------------------------
def pearson_huber_weight(c: float = PEARSON_C, sigma2: float = 1.0):
    r"""Classic robust-GLM weight ``w(r) = min(1, c/|r|)`` on the **Poisson Pearson
    residual** ``r = (y-μ)/√μ``, with ``μ`` from the quantile-projected Gaussian-log1p
    MAP.  Cell-wise (per ``i, j``).

    Why this μ (not the Poisson MAP): the ``log1p`` encoder *compresses* gross count
    outliers, so μ tracks the structure rather than the spike → ``r`` is large for an
    outlier → it is down-weighted.  A Poisson-Newton MAP would instead *chase* the
    outlier (``μ ≈ y`` → ``r ≈ 0`` → not flagged) and is unstable on contaminated data.

    Returns a ``weight_fn(gllvm, y) -> (n, p)`` for ``ZQEAutoFitter(weight_fn=…)``;
    the fitter calls it under no-grad and applies the *same* map to data and fantasies
    (so the centering supplies the Fisher-consistency correction automatically).
    """
    def weight_fn(gllvm, y):
        z = QuantileMapEncoder(gllvm, sigma2).forward(y)        # calibrated latent
        mu = gllvm.mean(z=z).clamp_min(1e-6)                    # Poisson conditional mean (n,p)
        r = (y.to(mu.dtype) - mu) / mu.sqrt()                   # Pearson residual (V=μ)
        return torch.clamp(c / r.abs().clamp_min(1e-12), max=1.0)
    return weight_fn


# ----------------------------------------------------------------------------
# Result-file bookkeeping
# ----------------------------------------------------------------------------
def _tag(eps: float, M: int) -> str:
    return f"e{int(round(eps * 1000)):03d}_M{int(M):06d}"


def result_path(eps: float, M: int, rep: int) -> str:
    return os.path.join(RESULTS_DIR,
                        f"q{Q}_p{P}_n{N}_{_tag(eps, M)}_rep{rep:03d}.csv")


def is_done(eps: float, M: int, rep: int) -> bool:
    return os.path.exists(result_path(eps, M, rep))


# ----------------------------------------------------------------------------
# Model / data / contamination
# ----------------------------------------------------------------------------
def fresh_decoder(device: str, T) -> GLLVM:
    g = GLLVM(latent_dim=Q, output_dim=P, bias=True).to(device)
    g.add_glm(PoissonGLM, idx=list(range(P)), params={"T": T}, name="P")
    with torch.no_grad():
        nn.init.normal_(g.wz, std=WZ_SCALE)
        nn.init.zeros_(g.bias)
    return g


def simulate_clean(seed: int):
    from gllvm.simulations import make_mixed, simulate
    torch.manual_seed(seed)
    g_true = make_mixed(n_latent=Q, poisson=P, wz_scale=WZ_SCALE)
    Y, _ = simulate(g_true, n_samples=N, device="cpu")
    W_true = g_true.wz.detach().to(torch.float64).numpy()
    b_true = g_true.bias.detach().to(torch.float64).numpy()
    return Y, W_true, b_true


def contaminate(Y: torch.Tensor, eps: float, M: int, seed: int) -> torch.Tensor:
    if eps <= 0 or M == 0:
        return Y
    Y = Y.clone()
    gen = torch.Generator().manual_seed(10_000 + seed)
    mask = torch.rand(Y.shape, generator=gen) < eps
    Y[mask] = float(M)
    return Y


# ----------------------------------------------------------------------------
# Estimators
# ----------------------------------------------------------------------------
def _align(W_true, W_est) -> np.ndarray:
    Wt = torch.as_tensor(np.asarray(W_true), dtype=torch.float64)
    We = (W_est.detach().to("cpu", torch.float64) if isinstance(W_est, torch.Tensor)
          else torch.as_tensor(np.asarray(W_est), dtype=torch.float64))
    return (We @ orthogonal_align(Wt, We)).numpy()


def _fit_zqe(Yc, seed, device, W_true, T, enc_factory):
    torch.manual_seed(seed)
    g = fresh_decoder(device, T)
    t0 = time.time()
    ft = ZQEAutoFitter(g, encoder_factory=enc_factory, device=device, seed=seed,
                       **ZQE_KW).fit(Yc.to(device))
    dt = time.time() - t0
    W = _align(W_true, ft.model.wz)
    b = ft.model.bias.detach().to("cpu", torch.float64).numpy()
    return W, b, dt, bool(ft.converged_)


def fit_zqe(Yc, seed, device, W_true):
    return _fit_zqe(Yc, seed, device, W_true, torch.log1p,
                    lambda g: MapEncoderGaussianLog1p(g))


def fit_zqe_huber(Yc, seed, device, W_true):
    c = huber_cut(Yc)
    return _fit_zqe(Yc, seed, device, W_true, make_T_huber(c),
                    lambda g, c=c: MapEncoderHuber(g, c))


def fit_zqe_pearson(Yc, seed, device, W_true, c: float = PEARSON_C):
    """EXPERIMENTAL — deferred to future work; **not in the paper sweep** (call with
    ``methods=("zqe_pearson",)`` to run it).  As of now it is *dominated* by
    ``zqe_huber``: it loses efficiency on clean data and climbs under contamination
    (the unweighted warm-up is still outlier-pulled → the weights, computed at a
    biased θ̂, mis-target; the de-shrunk quantile μ inflates the residual variance).
    The principled fix (a robust consistent start, à la Cantoni–Ronchetti) is left
    for the dedicated robustness paper.  Kept here as the starting point.

    Robust ZQE: **quantile (rank-projected) encoder** + per-observation
    **Pearson-Huber weight**.  Two reinforcing bounded-influence mechanisms:

    * **Robust ẑ (encoder).** The quantile encoder is rank-based per margin, so its
      output is *bounded* to the N(0,1) quantile range — a gross outlier can only
      become "the top rank" (ẑ ≈ Φ⁻¹(1−½/n)), never blow up → η is bounded → the
      statistic ``log1p(y)·η`` has bounded influence even before weighting.  (A plain
      ``log1p`` MAP leaks the outlier into ẑ; the rank projection caps it.)
    * **Pearson-Huber weight.** ``w(r)=min(1,c/|r|)``, ``r=(y-μ)/√μ``, μ from the same
      quantile latent; removes the outlier *cells* from the sum.  The two reinforce:
      a contaminated obs gets a too-high ẑ → its μ is off → its residuals are large →
      its cells are down-weighted.

    Warm-up runs **unweighted** (``weight_warmup=False``) for a consistent start; the
    weights act only in the gentle refine phase.  The model-fantasy term carries the
    same weight (each side weighted individually, on its own draws) → Fisher-consistency.
    """
    torch.manual_seed(seed)
    g = fresh_decoder(device, torch.log1p)
    t0 = time.time()
    ft = ZQEAutoFitter(g, encoder_factory=lambda g: QuantileMapEncoder(g),
                       weight_fn=pearson_huber_weight(c), device=device, seed=seed,
                       **ZQE_KW).fit(Yc.to(device))
    dt = time.time() - t0
    W = _align(W_true, ft.model.wz)
    b = ft.model.bias.detach().to("cpu", torch.float64).numpy()
    return W, b, dt, bool(ft.converged_)


def fit_gllvm(Yc, seed, rgllvm, W_true):
    t0 = time.time()
    rf = rgllvm.fit(Yc.cpu().numpy(), num_lv=Q, seed=seed)
    dt = time.time() - t0
    W = _align(W_true, rf.loadings)
    b = (np.asarray(rf.intercepts, dtype=float)
         if rf.intercepts is not None else np.full(W.shape[0], np.nan))
    return W, b, dt, float("nan")


# ----------------------------------------------------------------------------
# Record assembly (long / tidy: one row per scalar parameter)
# ----------------------------------------------------------------------------
def _rows(eps, M, rep, seed, method, W, b, time_sec, converged, procr, failed=0.0):
    rows = []
    common = dict(q=Q, p=P, n=N, eps=eps, M=M, rep=rep, seed=seed, method=method,
                  failed=failed, time_sec=time_sec, converged=converged,
                  procrustes=procr)
    W = None if W is None else np.asarray(W)
    for i in range(P):
        for j in range(Q):
            v = np.nan if W is None else float(W[i, j])
            rows.append({**common, "param": "W", "i": i, "j": j, "value": v})
    b = None if b is None else np.asarray(b)
    for i in range(P):
        v = np.nan if b is None else float(b[i])
        rows.append({**common, "param": "b", "i": i, "j": -1, "value": v})
    return rows


def run_condition(eps: float, M: int, rep: int, device: str, rgllvm: RGllvm,
                  methods=("zqe", "zqe_huber", "gllvm"),
                  overwrite: bool = False):
    """Fit one (condition, rep), **merging** with any existing result file.

    Only branches missing from the file are fitted (unless ``overwrite``), so a new
    branch (e.g. ``zqe_pearson``) is appended to finished reps without recomputing
    the others.  Returns the merged DataFrame, or ``None`` if nothing is needed.
    """
    path = result_path(eps, M, rep)
    existing = pd.read_csv(path) if os.path.exists(path) else None
    present = set(existing.method.unique()) if existing is not None else set()
    to_run = [m for m in methods if overwrite or m not in present]
    if not to_run and "true" in present:
        return None

    Y, W_true, b_true = simulate_clean(rep)
    Yc = contaminate(Y, eps, M, rep)

    rows = []
    if "true" not in present:
        rows += _rows(eps, M, rep, rep, "true", W_true, b_true,
                      np.nan, np.nan, np.nan, failed=np.nan)

    fitters = {
        "zqe": lambda: fit_zqe(Yc, rep, device, W_true),
        "zqe_huber": lambda: fit_zqe_huber(Yc, rep, device, W_true),
        "zqe_pearson": lambda: fit_zqe_pearson(Yc, rep, device, W_true),
        "gllvm": lambda: fit_gllvm(Yc, rep, rgllvm, W_true),
    }
    for m in to_run:
        try:
            W, b, dt, conv = fitters[m]()
            rows += _rows(eps, M, rep, rep, m, W, b, dt, conv,
                          procrustes_error(W_true, W), failed=0.0)
        except Exception as e:
            print(f"    !! {m} FAILED at eps={eps} M={M} rep{rep}: "
                  f"{type(e).__name__}: {e}")
            rows += _rows(eps, M, rep, rep, m, None, None,
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
def _present_methods(eps: float, M: int, rep: int) -> set:
    """Methods already stored in a rep's CSV (empty set if absent)."""
    path = result_path(eps, M, rep)
    if not os.path.exists(path):
        return set()
    return set(pd.read_csv(path, usecols=["method"]).method.unique())


def run_sweep(reps: int, *, device: Optional[str] = None,
              conditions=CONDITIONS,
              methods=("zqe", "zqe_huber", "gllvm"),
              overwrite: bool = False, rgllvm: Optional[RGllvm] = None,
              verbose: bool = True) -> None:
    """Run every (condition, rep) for ``rep`` in ``range(reps)``.

    Method-aware & resumable: a (condition, rep) runs only if some requested branch
    (or ``true``) is missing from its CSV; a new branch (``zqe_pearson``) is appended
    to finished reps without recomputing the others.  Run one new branch cheaply with
    ``methods=("zqe_pearson",)`` (no R needed unless ``gllvm`` is requested).
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if rgllvm is None and "gllvm" in methods:
        rgllvm = RGllvm(method="VA", family="poisson", timeout=60)
        if not rgllvm.available():
            raise RuntimeError(
                f"R Rscript not found at {rgllvm.rscript!r}; pass an RGllvm with "
                "rscript=.../workdir=..., or drop 'gllvm' from methods."
            )

    need = set(methods) | {"true"}
    todo = [(e, m, rep) for (e, m) in conditions for rep in range(reps)
            if overwrite or not need.issubset(_present_methods(e, m, rep))]
    if verbose:
        total = len(conditions) * reps
        print(f"device={device}  {len(conditions)} conditions × {reps} reps "
              f"= {total};  {total - len(todo)} complete, {len(todo)} to (re)run "
              f"[methods={','.join(methods)}].")

    for k, (eps, M, rep) in enumerate(todo, 1):
        t0 = time.time()
        df = run_condition(eps, M, rep, device, rgllvm, methods=methods,
                           overwrite=overwrite)
        if df is None:
            continue
        _save_atomic(df, result_path(eps, M, rep))
        if verbose:
            sub = df[df.method != "true"].drop_duplicates("method")
            tag = "  ".join(
                (f"{mm}:FAIL" if fl == 1.0 else f"{mm}:{pr:.3f}")
                for mm, fl, pr in zip(sub.method, sub.failed, sub.procrustes))
            print(f"[{k:>4}/{len(todo)}] eps={eps:<4} M={M:<6} rep{rep:03d}  "
                  f"{tag}  ({time.time()-t0:.0f}s)")


def load_results() -> pd.DataFrame:
    import glob
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, f"q{Q}_p{P}_n{N}_*.csv")))
    if not files:
        raise FileNotFoundError(f"no result CSVs in {RESULTS_DIR!r}; run the sweep first.")
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
