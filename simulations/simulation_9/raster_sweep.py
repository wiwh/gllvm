"""
Feasibility raster: ZQE-CPU, ZQE-GPU, and R gllvm each over a (q, p) grid at fixed n=500.

THREE methods, each run on EVERY cell with the SAME budget: 30-min (1800s) HARD cutoff + 15-min (900s)
SOFT budget (a cell finishing between SOFT and HARD is accepted but is the last p tried at that q):
  - zqe_gpu : ZQE on GPU       (fast at scale)
  - zqe_cpu : ZQE on CPU       (fair CPU-vs-gllvm comparison; slower at huge p)
  - gllvm   : R gllvm on CPU   (point fit; greys out where it times out/fails)

Both ZQE fits run in a TIMED SUBPROCESS (HARD cutoff TIMEOUT): caps runaway cells, isolates OOM/crashes,
and keeps resume robust (a cell over budget is marked 'slow' rather than re-run forever). gllvm uses
its own subprocess timeout (RGllvm).

One CSV row per (method,q,p): seconds, procW, status. Full W_hat saved per cell (truth regenerable:
highdim.make_data(p,q,500,seed=0)). Resumable: rerun skips done cells; gllvm monotonic early-stop.
Order: zqe_gpu -> gllvm -> zqe_cpu. Run: python raster_sweep.py
"""

import os, sys, time, csv, subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(
    0, os.path.join(_HERE, "..", "simulation_7")
)  # highdim = sim_7's ZQE recipe
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))  # gllvm package
import numpy as np
import torch
import highdim as H7
from gllvm.autofit import procrustes_error
from gllvm.r_gllvm import RGllvm
from gllvm.simulations import make_sparse, simulate

N = 500
Q_GRID = [1, 2, 3, 5, 8, 10, 15, 20]  # capped at 20: beyond it the row-sparsity cap means extra latents
# add no new info per response, so procW worsens for a reason orthogonal to scaling. q=20 is big enough.
P_GRID = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
TIMEOUT = 1800  # 30-min HARD per-cell cutoff (subprocess / R) for ALL methods
SOFT = 900  # 15-min SOFT budget: a cell over SOFT is still ACCEPTED (ok) but is the LAST p tried at
# that q (grey the rest) -> at most ONE >15min result per q, capturing the nonlinear blow-up gradually.
GPU = "cuda" if torch.cuda.is_available() else "cpu"
CSV = os.path.join(_HERE, "results", "raster.csv")
LOAD_DIR = os.path.join(_HERE, "results", "loadings")
COLS = ["method", "q", "p", "n", "seconds", "procW", "status"]
rgl = RGllvm(maxit=4000, timeout=TIMEOUT)
MAX_LV_PER_RESPONSE = (
    5  # cap latents affecting each response (row-sparsity) so eta = sum_k w_jk z_k
)
# stays bounded as q grows -> no exp(eta) saturation/overflow at large q


def make_data(p, q, n, seed=0):
    """Sparse Poisson GLLVM, each response affected by ~min(q, MAX_LV_PER_RESPONSE) latents.
    responses_per_latent set so E[latents/response] = q*rpl/p is capped (RASTER-LOCAL; keeps
    sim_7's highdim.make_data untouched). Truth regenerable from (p,q,seed)."""
    torch.manual_seed(seed)
    rpl = max(
        1, min(p, round(MAX_LV_PER_RESPONSE * p / q))
    )  # -> ~MAX latents per response
    g = make_sparse(
        n_latent=q,
        poisson=p,
        active_latent=q,
        wz_scale=H7.WZ_SCALE,
        responses_per_latent=rpl,
        lower_tri=H7.LOWER_TRI,
    )
    Y, _ = simulate(g, n_samples=n, device="cpu")
    return Y, g.wz.detach().cpu().clone()


def save_W(method, q, p, W):
    os.makedirs(LOAD_DIR, exist_ok=True)
    np.save(
        os.path.join(LOAD_DIR, f"{method}_q{q}_p{p}.npy"),
        np.asarray(W, dtype=np.float32),
    )


def load_done():
    done = {}
    if os.path.exists(CSV):
        with open(CSV) as f:
            for r in csv.DictReader(f):
                done[(r["method"], int(r["q"]), int(r["p"]))] = r
    return done


def is_break(rec):
    """Does this done cell end the p-staircase? True for any terminal/grey status, OR an 'ok' cell whose
    runtime exceeded the SOFT (15-min) budget -- it is kept, but no larger p is attempted at that q."""
    st = rec["status"]
    if st in ("slow", "oom", "fail", "timeout", "grey"):
        return True
    if st == "ok":
        try:
            return float(rec["seconds"]) > SOFT
        except (ValueError, TypeError):
            return False
    return False


def append(row):
    new = not os.path.exists(CSV)
    with open(CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new:
            w.writeheader()
        w.writerow(row)


def run_zqe(method, device, done):
    """ZQE on `device` for every cell, each in a timed subprocess (30-min HARD cutoff). Soft-budget
    staircase: while cells finish under SOFT (15min) keep going; the first cell that finishes BETWEEN
    SOFT and TIMEOUT is accepted (ok) but is the last p tried at that q; a cell over TIMEOUT (slow) or
    that OOMs greys the rest. Either way every larger p at that q is greyed WITHOUT running.
    """
    for q in Q_GRID:
        broken = False
        for p in P_GRID:
            rec = done.get((method, q, p))
            if rec is not None:
                if is_break(rec):
                    broken = True
                continue
            if broken:
                append(
                    dict(
                        method=method,
                        q=q,
                        p=p,
                        n=N,
                        seconds="",
                        procW="",
                        status="grey",
                    )
                )
                continue
            try:
                out = subprocess.run(
                    [
                        sys.executable,
                        os.path.abspath(__file__),
                        "zqefit",
                        str(p),
                        str(q),
                        device,
                        method,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=TIMEOUT,
                )
                line = next(
                    (l for l in out.stdout.splitlines() if l.startswith("ZFIT:")), None
                )
                if line is None:
                    oom = "out of memory" in out.stderr.lower()
                    append(
                        dict(
                            method=method,
                            q=q,
                            p=p,
                            n=N,
                            seconds="",
                            procW="",
                            status="oom" if oom else "fail",
                        )
                    )
                    print(
                        f"  {method} q={q:<3} p={p:<6} {'OOM' if oom else 'FAIL'}: "
                        f"{out.stderr.strip()[-80:]}",
                        flush=True,
                    )
                    if oom:
                        broken = True  # can't start -> larger p also can't -> grey rest
                    continue
                dt, pw = line[len("ZFIT:") :].split(",")
                dt = float(dt)
                append(
                    dict(
                        method=method,
                        q=q,
                        p=p,
                        n=N,
                        seconds=round(dt, 2),
                        procW=round(float(pw), 4),
                        status="ok",
                    )
                )
                if dt > SOFT:
                    broken = True  # over 15min: keep it, but it is the last p tried at this q
                    print(
                        f"  {method} q={q:<3} p={p:<6} {dt:7.0f}s  procW={float(pw):.3f}"
                        f"  (> {SOFT}s soft budget -> last p at q={q})",
                        flush=True,
                    )
                else:
                    print(
                        f"  {method} q={q:<3} p={p:<6} {dt:7.0f}s  procW={float(pw):.3f}",
                        flush=True,
                    )
            except subprocess.TimeoutExpired:
                append(
                    dict(
                        method=method,
                        q=q,
                        p=p,
                        n=N,
                        seconds=f">{TIMEOUT}",
                        procW="",
                        status="slow",
                    )
                )
                print(
                    f"  {method} q={q:<3} p={p:<6} SLOW (> {TIMEOUT}s = 30min) -> grey rest of q={q}",
                    flush=True,
                )
                broken = True  # time boundary hit -> larger p also -> grey rest


def run_gllvm(done):
    for q in Q_GRID:
        broken = False
        for p in P_GRID:
            rec = done.get(("gllvm", q, p))
            if rec is not None:
                if is_break(rec):
                    broken = True
                continue
            if broken:
                append(
                    dict(
                        method="gllvm",
                        q=q,
                        p=p,
                        n=N,
                        seconds="",
                        procW="",
                        status="grey",
                    )
                )
                continue
            Y, W_true = make_data(p, q, N, seed=0)
            t0 = time.time()
            try:
                fit = rgl.fit(Y.numpy(), num_lv=q, seed=1)
                dt = time.time() - t0
                pw = float(procrustes_error(W_true.numpy(), fit.loadings))
                save_W("gllvm", q, p, fit.loadings)
                append(
                    dict(
                        method="gllvm",
                        q=q,
                        p=p,
                        n=N,
                        seconds=round(dt, 1),
                        procW=round(pw, 4),
                        status="ok",
                    )
                )
                if dt > SOFT:
                    broken = True  # over 15min: keep it, but it is the last p tried at this q
                    print(
                        f"  gllvm   q={q:<3} p={p:<6} {dt:7.0f}s  procW={pw:.3f}"
                        f"  (> {SOFT}s soft budget -> last p at q={q})",
                        flush=True,
                    )
                else:
                    print(
                        f"  gllvm   q={q:<3} p={p:<6} {dt:7.0f}s  procW={pw:.3f}",
                        flush=True,
                    )
            except Exception as e:
                dt = time.time() - t0
                st = "timeout" if dt >= TIMEOUT - 8 else "fail"
                append(
                    dict(
                        method="gllvm",
                        q=q,
                        p=p,
                        n=N,
                        seconds=round(dt, 1),
                        procW="",
                        status=st,
                    )
                )
                print(
                    f"  gllvm   q={q:<3} p={p:<6} {st.upper()} ({dt:.0f}s) -> grey rest of q={q}",
                    flush=True,
                )
                broken = True


def zqefit_cli(p, q, device, method):
    """Single ZQE fit on `device`; save loadings; print 'ZFIT:seconds,procW' (fit time only)."""
    Y, W_true = make_data(p, q, N, seed=0)
    ft, dt, pw = H7.fit(Y, q, l2=H7.L2_COEF / N, device=device, seed=0, W_true=W_true)
    save_W(method, q, p, ft.model.wz.detach().cpu().numpy())
    print(f"ZFIT:{dt},{float(pw)}", flush=True)


def main():
    os.makedirs(os.path.join(_HERE, "results"), exist_ok=True)
    print(
        f"GPU={GPU}  n={N}  HARD cutoff={TIMEOUT}s (30min), SOFT budget={SOFT}s (15min) for ALL methods\n"
        f"grid q={Q_GRID} x p={P_GRID}",
        flush=True,
    )
    print("[pass 1/3] zqe_gpu", flush=True)
    run_zqe("zqe_gpu", GPU, load_done())
    print("[pass 2/3] gllvm", flush=True)
    run_gllvm(load_done())
    print("[pass 3/3] zqe_cpu", flush=True)
    run_zqe("zqe_cpu", "cpu", load_done())
    print("\ndone -> results/raster.csv", flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "zqefit":
        zqefit_cli(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5])
    else:
        main()
