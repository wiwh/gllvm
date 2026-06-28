"""
Does R gllvm's OWN deterministic CI (Laplace/VA observed-information SE) conform to the bootstrap?

For each dataset we fit gllvm(num.lv=2, poisson, sd.errors=TRUE), pull its loadings W_g and their
SEs (sd.theta scaled by sigma.lv). Compare:
  - point fit sanity:  procrustes error of W_g vs truth (should be ~0.15, like ZQE),
  - SE scale:          median/IQR of gllvm SE vs the ZQE bootstrap SE (validation.npz), per dataset,
  - self-calibration:  mean gllvm SE vs the empirical sampling SD of gllvm loadings across datasets
                       (gauge-aligned), i.e. does gllvm's Laplace SE actually calibrate?

Gauge note: per-entry SE lives in each fit's gauge, so the SE-vs-SD calibration is done on the
Frobenius scale (gauge-invariant): mean trace(SE^2) vs trace(empirical Var) of aligned loadings.

Run: python gllvm_se_check.py
"""
import os, sys, subprocess, textwrap, time
sys.path.insert(0, os.path.abspath("../../src"))
import numpy as np
import varboot
from gllvm.r_gllvm import RGllvm, _wsl_to_win

P, Q, N, WZ = 50, 2, 500, 0.5
N_FIT = 10
HERE = os.path.dirname(os.path.abspath(__file__))
_r = RGllvm()                                   # use its Windows-accessible workdir (avoids UNC)
WORK = os.path.join(_r.workdir, "se_check")
os.makedirs(WORK, exist_ok=True)
RSCRIPT = _r.rscript


def gllvm_fit_se(Y, tag):
    yf = os.path.join(WORK, f"Y_{tag}.csv")
    wf = os.path.join(WORK, f"W_{tag}.csv")
    sf = os.path.join(WORK, f"SE_{tag}.csv")
    rf = os.path.join(WORK, f"run_{tag}.R")
    np.savetxt(yf, np.rint(Y).astype(int), delimiter=",", fmt="%d")
    for stale in (wf, sf):
        if os.path.exists(stale):
            os.remove(stale)
    script = textwrap.dedent(f"""
        Y <- as.matrix(read.csv("{_wsl_to_win(yf)}", header=FALSE))
        suppressPackageStartupMessages(library(gllvm))
        fit <- gllvm(Y, num.lv={Q}, family="poisson", method="VA", sd.errors=TRUE,
                     control=list(TMB=TRUE, maxit=6000, trace=FALSE))
        sig <- fit$params$sigma.lv
        W   <- sweep(fit$params$theta, 2, sig, "*")
        seW <- sweep(fit$sd$theta,     2, sig, "*")
        write.csv(W,   "{_wsl_to_win(wf)}", row.names=FALSE)
        write.csv(seW, "{_wsl_to_win(sf)}", row.names=FALSE)
        cat("done\\n")
    """).strip()
    with open(rf, "w") as f:
        f.write(script)
    p = subprocess.run([RSCRIPT, "--vanilla", _wsl_to_win(rf)],
                       capture_output=True, text=True, timeout=900)
    if not os.path.exists(wf):
        raise RuntimeError(f"gllvm failed:\n{p.stderr[-1500:]}")
    W = np.loadtxt(wf, delimiter=",", skiprows=1).reshape(P, Q)
    seW = np.loadtxt(sf, delimiter=",", skiprows=1).reshape(P, Q)
    return W, seW


def procr_err(Wt, We):
    U, _, Vt = np.linalg.svd(We.T @ Wt)
    R = U @ Vt
    return np.linalg.norm(We @ R - Wt) / np.linalg.norm(Wt), R


def main():
    truth = varboot.make_truth(P, Q, wz_scale=WZ, seed=0, responses_per_latent=P)
    W_true = truth.wz.detach().cpu().numpy()
    d = np.load(os.path.join(HERE, "results", "validation.npz"))
    Wboot, mask = d["Wboot"], d["mask"].astype(bool)

    Ws, SEs = [], []
    print(f"fitting gllvm (sd.errors) on {N_FIT} datasets ...", flush=True)
    for dd in range(N_FIT):
        t0 = time.time()
        Y = varboot.sample_data(truth, N, seed=1000 + dd).numpy()
        try:
            W, seW = gllvm_fit_se(Y, f"d{dd}")
        except Exception as e:
            print(f"[{dd}] FAILED: {str(e)[:120]}", flush=True); continue
        pe, _ = procr_err(W_true, W)
        se_boot = Wboot[dd][:, mask].std(0, ddof=1)
        se_g = seW[mask]
        print(f"[{dd}] procW_gllvm={pe:.3f}  gllvm SE med={np.median(se_g):.4f} "
              f"[{np.percentile(se_g,10):.4f},{np.percentile(se_g,90):.4f}]  "
              f"boot SE med={np.median(se_boot):.4f}  ratio(med)={np.median(se_g)/np.median(se_boot):.2f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        Ws.append(W); SEs.append(seW)

    Ws = np.stack(Ws); SEs = np.stack(SEs)
    # self-calibration on the gauge-invariant Frobenius scale
    Wa = np.stack([w @ procr_err(W_true, w)[1] for w in Ws])     # align to truth
    emp_var_fro = ((Wa - Wa.mean(0)) ** 2).sum(1).mean()         # mean ||.||^2 spread per row...
    emp_sd_fro = np.sqrt(((Wa - Wa.mean(0)) ** 2).sum() / (len(Wa) - 1))
    mean_se_fro = np.sqrt((SEs ** 2).sum(axis=(1, 2)).mean())
    print(f"\ngllvm self-calibration (Frobenius, gauge-free): "
          f"empirical SD={emp_sd_fro:.3f}  mean det-SE={mean_se_fro:.3f}  "
          f"ratio={mean_se_fro/emp_sd_fro:.2f}")
    np.savez(os.path.join(HERE, "results", "gllvm_se.npz"), Ws=Ws, SEs=SEs, W_true=W_true)


if __name__ == "__main__":
    main()
