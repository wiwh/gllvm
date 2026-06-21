"""
sweep.py — driver for *simulation_6* (encoder transfer: frozen population encoder).

The transfer experiment, ZQE-only. For the Gaussian-MAP encoder the "encoder" is
just the decoder parameters (W, b): `z = (WᵀW+σ²I)⁻¹ Wᵀ(log1p(y)−b)`. So:

1. **Population**: fit ZQE once on a *large* sample from the true model θ₀ → θ_pop
   (= the population-trained encoder). Done a single time, reused for all reps.
2. **Small samples**: take the true model, **shift one loading** `W[i*,j*]` across a
   grid (−3…3). For each grid value, draw a *small* sample and fit ZQE two ways:
     - **frozen**:   encoder held at θ_pop (does NOT track the decoder),
     - **unfrozen**: ordinary ZQE (encoder tracks the decoder; "sample-only").
3. **Expectations**: (a) the frozen encoder still **consistently recovers** the
   shifted loading across the grid (score-function identity: any fixed encoder is a
   valid mechanism); (b) it **gains** over sample-only, because θ_pop is estimated
   from the large sample, so the frozen encoder carries low-variance information
   about the *other* loadings.

Pure ZQE, existing src only (frozen encoder = `MapEncoderGaussianLog1p` bound to a
fixed copy of the population decoder). Results: `results/transfer.csv`.
"""

from __future__ import annotations

import copy
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from gllvm.gllvm_module import GLLVM
from gllvm.glms import PoissonGLM
from gllvm.encoder import MapEncoderGaussianLog1p
from gllvm.autofit import ZQEAutoFitter, orthogonal_align, procrustes_error
from gllvm.simulations import make_mixed, simulate

Q, P = 2, 20
WZ_SCALE = 1.0
N_POP = 5000           # large population sample (encoder trained here, once)
N_SAMPLE = 50          # small target sample
ISTAR, JSTAR = 0, 0    # which loading entry is shifted
GRID = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

ZQE_KW = dict(steps_per_round=150, max_rounds=10, tol=0.001,
              refine_lr=0.5, warmup_lr=0.5, ema_decay=0.95, verbose=False)


def true_model(seed: int = 0) -> GLLVM:
    """The base true model θ₀ (CPU)."""
    torch.manual_seed(seed)
    return make_mixed(n_latent=Q, poisson=P, wz_scale=WZ_SCALE)


def fresh_decoder(device: str) -> GLLVM:
    g = GLLVM(latent_dim=Q, output_dim=P, bias=True).to(device)
    g.add_glm(PoissonGLM, idx=list(range(P)), params={"T": torch.log1p}, name="P")
    with torch.no_grad():
        nn.init.normal_(g.wz, std=WZ_SCALE)
        nn.init.zeros_(g.bias)
    return g


def fit_zqe(Y, device, encoder_factory, seed):
    """Fit ZQE on Y with a given encoder_factory; returns the Polyak-averaged decoder."""
    torch.manual_seed(seed)
    g = fresh_decoder(device)
    ft = ZQEAutoFitter(g, encoder_factory=encoder_factory, device=device,
                       seed=seed, **ZQE_KW).fit(Y.to(device))
    return ft.model


def fit_population(g_true, device, seed=0):
    """Fit ZQE on a large sample from θ₀; return a frozen copy = the population encoder."""
    Y, _ = simulate(g_true, n_samples=N_POP, device="cpu")
    g_pop = fit_zqe(Y, device, lambda g: MapEncoderGaussianLog1p(g), seed)
    return copy.deepcopy(g_pop).to(device)


def _aligned_entry(W_true, W_est, i, j) -> float:
    """Estimated entry (i,j) after Procrustes-aligning the estimate to the (shifted) truth."""
    Wt = torch.as_tensor(np.asarray(W_true), dtype=torch.float64)
    We = (W_est.detach().to("cpu", torch.float64) if isinstance(W_est, torch.Tensor)
          else torch.as_tensor(np.asarray(W_est), dtype=torch.float64))
    return float((We @ orthogonal_align(Wt, We))[i, j])


def run_sweep(reps: int = 10, *, grid=GRID, device=None, g_true=None, g_pop=None,
              verbose: bool = True):
    """Population-encoder transfer sweep. Fits the population once (unless ``g_pop``
    is supplied), then loops over the loading grid × reps, fitting frozen vs unfrozen.
    Returns (DataFrame, g_true, g_pop)."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if g_true is None:
        g_true = true_model(seed=0)
    v0 = float(g_true.wz[ISTAR, JSTAR])
    if g_pop is None:
        t0 = time.time()
        g_pop = fit_population(g_true, device, seed=0)
        if verbose:
            print(f"[population] n={N_POP} fit once: "
                  f"Procrustes(θpop,θ0)={procrustes_error(g_true.wz, g_pop.wz):.3f}  "
                  f"true W[{ISTAR},{JSTAR}]=v0={v0:.2f}  ({time.time()-t0:.0f}s)")
    g_pop_frozen = copy.deepcopy(g_pop).to(device)

    rows = []
    todo = [(v, rep) for v in grid for rep in range(reps)]
    for k, (v, rep) in enumerate(todo, 1):
        gt = copy.deepcopy(g_true)
        with torch.no_grad():
            gt.wz[ISTAR, JSTAR] = float(v)          # shift one loading
        seed = 1000 + rep
        torch.manual_seed(seed)
        Y, _ = simulate(gt, n_samples=N_SAMPLE, device="cpu")
        Wtgt = gt.wz.detach().cpu().numpy()

        gf = fit_zqe(Y, device, lambda g: MapEncoderGaussianLog1p(g_pop_frozen), seed)
        gu = fit_zqe(Y, device, lambda g: MapEncoderGaussianLog1p(g), seed)
        for name, gh in [("frozen", gf), ("unfrozen", gu)]:
            rows.append(dict(v=float(v), rep=rep, method=name, seed=seed,
                             est_entry=_aligned_entry(Wtgt, gh.wz, ISTAR, JSTAR),
                             procrustes=procrustes_error(Wtgt, gh.wz)))
        if verbose:
            sub = {r["method"]: r for r in rows[-2:]}
            print(f"[{k:>3}/{len(todo)}] v={v:+.1f} rep{rep:02d}  "
                  f"frozen:est={sub['frozen']['est_entry']:+.2f}/P={sub['frozen']['procrustes']:.3f}  "
                  f"unfrozen:est={sub['unfrozen']['est_entry']:+.2f}/P={sub['unfrozen']['procrustes']:.3f}")

    df = pd.DataFrame(rows)
    df.attrs["v0"] = v0
    df.to_csv(os.path.join(RESULTS_DIR, "transfer.csv"), index=False)
    return df, g_true, g_pop
