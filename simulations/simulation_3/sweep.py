"""
sweep.py — driver for *simulation_3* (Binomial GLLVM size sweep).

The **symmetric twin of simulation 1**: the identical size sweep
(``q=2``; ``p ∈ {10,20,50,100}``; ``n ∈ {20,100,500}``; ``H`` reps), but every
response is **Binomial** (``N=10`` trials, logit link) instead of Poisson — a
bounded-count twin of the Poisson (unbounded-count) sweep, on *clean* data.
(Robustness to contamination lives only in simulation 2.)

**Why Binomial with N>1 and not Bernoulli.** With a single trial (Bernoulli),
the GLLVM likelihood is *unbounded* at small n: the loadings can drive logits to
±∞ and perfectly separate the 0/1 outcomes, so the parameter is **not
identifiable** — even at ``wz_scale=0.5`` (where the *population* probabilities
are mid-range, ≈83–97% in [.15,.85]) the *empirical* n=20 outcomes still separate
and the ZQE/gllvm estimates run away (Procrustes ≈ 2–3, verified). With
``N = BINOM_TRIALS`` trials the counts cannot be perfectly separated, the
likelihood is bounded, and the setting is identifiable across the whole grid. This
is an identifiability fix in the *design*, not a regulariser on the estimator.

Two things change vs simulation 1:

* **Statistic.** Poisson used ``T=log1p``. Here the response is a bounded count,
  so we use the centred-and-scaled proportion ``T(y) = 4·(y/N − 0.5)`` ∈ [-2, 2].
  Any measurable ``T`` is valid (score-function identity).
* **Encoder.** The Gaussian-proxy MAP encoder solves the same ridge system on
  ``T(y)`` (:class:`MapEncoderGaussianT`). Encoder affects efficiency, not consistency.

Loadings use ``wz_scale=0.5`` (matching simulation 1) so success probabilities
stay mixed around 0.5, and ZQE carries the same ridge ``l2 = 0.5/n``.
Everything else matches simulation 1: ``seed == rep`` drives the true
model and data; one CSV per (setting, rep); resumable; failures flagged; loadings
scored vs the true ``W`` after Procrustes rotation. ``gllvm`` is fit with
``family="binomial"`` and ``link="logit"`` (its binomial default is *probit*;
without this the recovered loadings are off by the logit/probit scale ~1.8).
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
from gllvm.glms import BinomialGLM
from gllvm.autofit import ZQEAutoFitter, orthogonal_align, procrustes_error
from gllvm.r_gllvm import RGllvm

# ----------------------------------------------------------------------------
# Sweep grid — identical to simulation 1
# ----------------------------------------------------------------------------
Q = 2
P_GRID = [10, 20, 50, 100]
N_GRID = [20, 100, 500]
WZ_SCALE = 0.5             # loading scale (matches simulation 1)
L2_COEF = 0.5              # ridge coefficient; per-fit penalty is L2_COEF / n
BINOM_TRIALS = 10          # Binomial trials per cell. N=1 (Bernoulli) is NOT
#                            identifiable at small n (perfect separation → loadings
#                            run away even at wz_scale=0.5: empirical 0/1 outcomes
#                            separate when n is small); N>1 bounds the likelihood.
#                            N=10 is cleanly identifiable across the whole grid.

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

ZQE_KW = dict(steps_per_round=150, max_rounds=10, tol=0.001,
              refine_lr=0.5, warmup_lr=0.5, ema_decay=0.95, verbose=False)


# ZQE statistic for binomial (module-level → deepcopy-stable inside the fitter):
# centred & scaled proportion, T(y) = 4·(y/N − 0.5) ∈ [-2, 2].
def T_binom(y):
    return 4.0 * (y.float() / BINOM_TRIALS - 0.5)


# ----------------------------------------------------------------------------
# Gaussian-proxy MAP encoder on a configurable statistic T(y)
# ----------------------------------------------------------------------------
class MapEncoderGaussianT(nn.Module):
    """Parameter-free MAP encoder ``z = (sigma2 I + WᵀW)⁻¹ Wᵀ (T(y) − b)``.

    Same closed form as ``MapEncoderGaussianLog1p`` but on an arbitrary response
    statistic ``transform`` (here ``T_binary``); holds a live reference to the
    decoder so it tracks the current ``W, b``.
    """

    def __init__(self, gllvm, transform, sigma2: float = 1.0):
        super().__init__()
        self.gllvm = gllvm
        self.transform = transform
        self.sigma2 = sigma2

    def forward(self, y):
        W = self.gllvm.wz
        b = (self.gllvm.bias if self.gllvm.bias is not None
             else torch.zeros(W.shape[0], device=W.device, dtype=W.dtype))
        t_y = self.transform(y).to(W.dtype)
        rhs = (t_y - b.unsqueeze(0)) @ W
        A = (self.sigma2 * torch.eye(W.shape[1], device=W.device, dtype=W.dtype)
             + W.T @ W)
        return torch.linalg.solve(A, rhs.T).T

    def sample(self, y):
        z = self.forward(y)
        return z, z, torch.full_like(z, float("-inf"))


# ----------------------------------------------------------------------------
# Result-file bookkeeping (identical scheme to simulation 1)
# ----------------------------------------------------------------------------
def result_path(q: int, p: int, n: int, rep: int) -> str:
    return os.path.join(RESULTS_DIR, f"q{q}_p{p}_n{n}_rep{rep:03d}.csv")


def is_done(q: int, p: int, n: int, rep: int) -> bool:
    return os.path.exists(result_path(q, p, n, rep))


# ----------------------------------------------------------------------------
# Model / data
# ----------------------------------------------------------------------------
def fresh_decoder(q: int, p: int, device: str) -> GLLVM:
    g = GLLVM(latent_dim=q, output_dim=p, bias=True).to(device)
    g.add_glm(BinomialGLM, idx=list(range(p)),
              params={"total_count": BINOM_TRIALS, "T": T_binom}, name="B")
    with torch.no_grad():
        nn.init.normal_(g.wz, std=WZ_SCALE)
        nn.init.zeros_(g.bias)
    return g


def simulate_setting(q: int, p: int, n: int, seed: int):
    """Clean dense Bernoulli model + dataset on CPU. Returns ``(Y, W_true, b_true)``."""
    from gllvm.simulations import make_mixed, simulate
    torch.manual_seed(seed)
    g_true = make_mixed(n_latent=q, binomial=p, binom_trials=BINOM_TRIALS, wz_scale=WZ_SCALE)
    Y, _ = simulate(g_true, n_samples=n, device="cpu")
    W_true = g_true.wz.detach().to(torch.float64).numpy()
    b_true = g_true.bias.detach().to(torch.float64).numpy()
    return Y, W_true, b_true


# ----------------------------------------------------------------------------
# Estimators
# ----------------------------------------------------------------------------
def _align(W_true, W_est) -> np.ndarray:
    Wt = torch.as_tensor(np.asarray(W_true), dtype=torch.float64)
    We = (W_est.detach().to("cpu", torch.float64) if isinstance(W_est, torch.Tensor)
          else torch.as_tensor(np.asarray(W_est), dtype=torch.float64))
    return (We @ orthogonal_align(Wt, We)).numpy()


def fit_zqe(Y, q, p, seed, device, W_true):
    torch.manual_seed(seed)
    g = fresh_decoder(q, p, device)
    t0 = time.time()
    ft = ZQEAutoFitter(g, encoder_factory=lambda g: MapEncoderGaussianT(g, T_binom),
                       device=device, seed=seed, l2=L2_COEF / Y.shape[0],
                       **ZQE_KW).fit(Y.to(device))
    dt = time.time() - t0
    W = _align(W_true, ft.model.wz)
    b = ft.model.bias.detach().to("cpu", torch.float64).numpy()
    return W, b, dt, bool(ft.converged_)


def fit_gllvm(Y, q, seed, rgllvm, W_true):
    t0 = time.time()
    rf = rgllvm.fit(Y.cpu().numpy(), num_lv=q, seed=seed)
    dt = time.time() - t0
    W = _align(W_true, rf.loadings)
    b = (np.asarray(rf.intercepts, dtype=float)
         if rf.intercepts is not None else np.full(W.shape[0], np.nan))
    return W, b, dt, float("nan")


# ----------------------------------------------------------------------------
# Record assembly (long / tidy — identical schema to simulation 1)
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
                rgllvm: RGllvm, methods=("zqe", "gllvm")) -> pd.DataFrame:
    """Fit one (setting, rep): simulate clean binary, fit each method."""
    Y, W_true, b_true = simulate_setting(q, p, n, seed)
    rows = _rows(q, p, n, seed, seed, "true", W_true, b_true,
                 np.nan, np.nan, np.nan, failed=np.nan)
    fitters = {"zqe": lambda: fit_zqe(Y, q, p, seed, device, W_true),
               "gllvm": lambda: fit_gllvm(Y, q, seed, rgllvm, W_true)}
    for m in methods:
        try:
            W, b, dt, conv = fitters[m]()
            rows += _rows(q, p, n, seed, seed, m, W, b, dt, conv,
                          procrustes_error(W_true, W), failed=0.0)
        except Exception as e:
            print(f"    !! {m} FAILED at q{q} p{p} n{n} rep{seed}: "
                  f"{type(e).__name__}: {e}")
            rows += _rows(q, p, n, seed, seed, m, None, None,
                          np.nan, np.nan, np.nan, failed=1.0)
    return pd.DataFrame(rows)


def _save_atomic(df: pd.DataFrame, path: str) -> None:
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------
# Sweep driver
# ----------------------------------------------------------------------------
def run_sweep(reps: int, *, q: int = Q, p_grid=P_GRID, n_grid=N_GRID,
              device: Optional[str] = None, methods=("zqe", "gllvm"),
              overwrite: bool = False, rgllvm: Optional[RGllvm] = None,
              verbose: bool = True) -> None:
    """Run the full grid for ``rep`` in ``range(reps)``, skipping finished reps."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if rgllvm is None and "gllvm" in methods:
        # link="logit" is ESSENTIAL: gllvm's binomial default is probit, which
        # would report loadings on a different scale (~1.8× = π/√3) than the
        # logit-generated truth and make the comparison unfair.
        rgllvm = RGllvm(method="VA", family="binomial",
                        ntrials=BINOM_TRIALS, link="logit")
        if not rgllvm.available():
            raise RuntimeError(
                f"R Rscript not found at {rgllvm.rscript!r}; pass an RGllvm with "
                "rscript=.../workdir=..., or drop 'gllvm' from methods."
            )

    settings = [(p, n) for p in p_grid for n in n_grid]
    todo = [(p, n, rep) for (p, n) in settings for rep in range(reps)
            if overwrite or not is_done(q, p, n, rep)]
    if verbose:
        total = len(settings) * reps
        print(f"device={device}  grid={len(settings)} settings × {reps} reps "
              f"= {total} fits;  {total - len(todo)} done, {len(todo)} to run.")

    for k, (p, n, rep) in enumerate(todo, 1):
        t0 = time.time()
        df = run_setting(q, p, n, rep, device, rgllvm, methods=methods)
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
    """Concatenate every per-rep CSV under ``results/``."""
    import glob
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, f"q{q}_p*_n*_rep*.csv")))
    if not files:
        raise FileNotFoundError(f"no result CSVs in {RESULTS_DIR!r}; run the sweep first.")
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
