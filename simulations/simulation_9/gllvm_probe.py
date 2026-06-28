"""
Probe: where does R gllvm's wall-time explode / time out? Find the break point on the (p, q) grid
so we can design the scaling figure that bridges simulation_1 -> simulation_7.

Times gllvm POINT estimation (VA, no SE) at n=1000 across a grid, with a timeout. Prints seconds or
TIMEOUT/FAIL. We just want the break locus (gllvm cost ~ driven by p*q).

Run: python gllvm_probe.py
"""
import os, sys, time
sys.path.insert(0, os.path.abspath("../../src"))
import numpy as np
import torch
from gllvm.simulations import make_sparse, simulate
from gllvm.r_gllvm import RGllvm

N = 1000
WZ = 0.5
TIMEOUT = 420                       # seconds per fit; treat longer as "broken"
GRID = [(100, 2), (100, 5), (100, 10),
        (300, 2), (300, 5), (300, 10),
        (1000, 2), (1000, 5)]

r = RGllvm(maxit=4000, timeout=TIMEOUT)


def make_Y(p, q, seed=0):
    torch.manual_seed(seed)
    g = make_sparse(n_latent=q, poisson=p, active_latent=q, wz_scale=WZ,
                    responses_per_latent=max(2, p // 2), lower_tri=True)
    Y, _ = simulate(g, n_samples=N, device="cpu")
    return Y.numpy()


def main():
    print(f"gllvm point-fit timing  (n={N}, timeout={TIMEOUT}s)\n  p     q    seconds", flush=True)
    for p, q in GRID:
        Y = make_Y(p, q)
        t0 = time.time()
        try:
            r.fit(Y, num_lv=q, seed=1)
            dt = time.time() - t0
            print(f"  {p:<5} {q:<4} {dt:7.0f}", flush=True)
        except Exception as e:
            dt = time.time() - t0
            tag = "TIMEOUT" if dt >= TIMEOUT - 5 else "FAIL"
            print(f"  {p:<5} {q:<4} {tag}  ({dt:.0f}s)  {str(e)[:80]}", flush=True)


if __name__ == "__main__":
    main()
