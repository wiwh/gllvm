"""
simulation_8 — generate + store the loading-SE validation data.

For D datasets from a FIXED truth, fit theta_hat once per dataset, then bootstrap the loadings
with the validated CRN parametric bootstrap that re-solves the actual estimator
(``param_bootstrap_resolve``: Poisson-MAP encoder, T=log1p, fixed fantasy seed).

We store the RAW loading matrices — true W, per-dataset point estimate Whats, and the full
bootstrap stacks Wboot — so any functional (lower-triangular W entries, Gram (WW^T)_jk, ...) can
be scored afterwards in make_figure.py WITHOUT rerunning. Run: ``python run_validation.py``.
"""
import os, sys, time
sys.path.insert(0, os.path.abspath("../../src"))
import numpy as np
import torch
import varboot
from gllvm.autofit import procrustes_error

# Settings match simulation_1's q=2 dense sweep (the headline sims): the well-behaved large-n
# cell p=50, n=500, q=2 (lowest procW ~0.15, stable -> the right regime for a coverage study).
P, Q, N = 50, 2, 500
L2_COEF, WZ = 0.001, 0.5
RESP_PER_LATENT = P            # dense loadings (responses_per_latent=p), as in simulation_1
D, H = 40, 120                 # overnight run (~10h); per-dataset checkpointed
DEV = "cuda" if torch.cuda.is_available() else "cpu"
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)


def _resume(ckpt, W_true):
    """Load a matching checkpoint to resume from. Returns (Whats, Wboot, procs, mask, start)
    or fresh empties if no usable checkpoint. Resume is exact because every dataset is fully
    seed-deterministic (seed=1000+d, fantasy_seed=999)."""
    if not os.path.exists(ckpt):
        return [], [], [], None, 0
    z = np.load(ckpt)
    same = (int(z["p"]) == P and int(z["q"]) == Q and int(z["n"]) == N and int(z["H"]) == H
            and z["Whats"].shape[1:] == tuple(W_true.shape)
            and np.allclose(z["W_true"], W_true.numpy()))
    if not same:
        print("checkpoint config mismatch -> starting fresh", flush=True)
        return [], [], [], None, 0
    done = int(z["D"])
    print(f"resuming from checkpoint: {done}/{D} datasets already done", flush=True)
    Whats = [torch.from_numpy(z["Whats"][i]) for i in range(done)]
    Wboot = [torch.from_numpy(z["Wboot"][i]) for i in range(done)]
    procs = list(z["procW"])
    return Whats, Wboot, procs, torch.from_numpy(z["mask"]), done


def main():
    print(f"device={DEV}  p={P} q={Q} n={N} (dense)  D={D}  H={H}", flush=True)
    truth = varboot.make_truth(P, Q, wz_scale=WZ, seed=0, responses_per_latent=RESP_PER_LATENT)
    W_true = truth.wz.detach().cpu().clone()                       # (p, q)

    ckpt = os.path.join(HERE, "results", "validation.npz")
    Whats, Wboot, procs, mask, start = _resume(ckpt, W_true)
    if start >= D:
        print(f"already complete ({start}/{D}); nothing to do", flush=True)
        return
    for d in range(start, D):
        t0 = time.time()
        Y = varboot.sample_data(truth, N, seed=1000 + d)
        ft = varboot.fit_point(Y, Q, l2=L2_COEF / N, device=DEV, seed=1000 + d, wz_scale=WZ)
        if mask is None:
            mask = ft.model.wz_mask.detach().cpu().clone()        # (p, q) free-entry mask
        Wd = ft.model.wz.detach().cpu().clone()
        Wb = varboot.param_bootstrap_resolve(ft, H, device=DEV, l2=L2_COEF / N,
                                             seed=8000 + d, fantasy_seed=999)   # (H, p, q)
        Whats.append(Wd)
        Wboot.append(Wb)
        procs.append(float(procrustes_error(W_true, Wd)))
        print(f"[{d+1:>2}/{D}] procW={procs[-1]:.3f}  ({time.time()-t0:.0f}s)", flush=True)
        # checkpoint after every dataset: partial results are visible + resumable
        np.savez(ckpt,
                 W_true=W_true.numpy(), Whats=torch.stack(Whats).numpy(),
                 Wboot=torch.stack(Wboot).numpy(), mask=mask.numpy(),
                 p=P, q=Q, n=N, D=d + 1, H=H, procW=np.array(procs))

    print(f"\nsaved results/validation.npz  ({len(Whats)} datasets, Wboot per-dataset {Wb.shape})",
          flush=True)


if __name__ == "__main__":
    main()
