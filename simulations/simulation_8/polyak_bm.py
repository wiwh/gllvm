"""
Prototype: "free" inference from the Polyak-averaged SGD path via BATCH-MEANS.

Idea (Chen-Lee-Tong-Zhang 2020; Polyak-Ruppert): warm-start at theta_hat, run ONE constant-LR
SGD chain (the existing ZQEAutoFitter refine chain, max_rounds=1, refine_lr_power=0 -> stationary
process around theta_hat). The per-step gradient noise is the fantasy redraw, whose covariance
equals the data score covariance S at the root (Fisher consistency). The AVERAGED iterate's
long-run variance is the sandwich A^-1 S A^-T; batch-means estimates it from the path autocorrelation
(NOT the marginal iterate scatter, which is the O(lr) Lyapunov matrix).

Validation: compare batch-means SE to the CRN bootstrap SE in results/validation.npz (the ground
truth, calibrated Wald). Success = SE_bm proportional to SE_boot across functionals (single
calibration constant c). We score the lower-tri free loadings (identified, no alignment needed
within a single chain anchored at theta_hat).

Run: python polyak_bm.py
"""
import os, sys, time
sys.path.insert(0, os.path.abspath("../../src"))
import numpy as np
import torch
import varboot
from gllvm.gllvm_module import GLLVM
from gllvm.glms import PoissonGLM
from gllvm.encoder import MapEncoderPoissonNewton
from gllvm.autofit import ZQEAutoFitter

P, Q, N = 50, 2, 500
L2, WZ = 0.001, 0.5
DEV = "cuda" if torch.cuda.is_available() else "cpu"
HERE = os.path.dirname(os.path.abspath(__file__))

N_TEST = 4            # datasets to check (subset of the 40)
K = 3000             # Polyak chain length (single SGD steps)
BURN = 0.3           # fraction discarded before averaging
N_BATCH = 30         # non-overlapping batches for batch-means
LR = 0.05            # constant "wiggle" LR
N_Q = N              # fantasy samples per step (sets noise scale -> calibration constant)


def polyak_chain(Y, theta_w, theta_b, seed):
    """ONE constant-LR Polyak chain, **single SGD steps**, warm-started at (theta_w, theta_b).
    Full real data (deterministic m1); ONE fresh fantasy draw per step (random m2) -> per-step
    noise = score covariance S (Fisher consistency). Logs every iterate. Returns (K, p, q) CPU."""
    g = GLLVM(latent_dim=Q, output_dim=P, bias=True, lower_tri=True).to(DEV)
    g.add_glm(PoissonGLM, idx=list(range(P)), params={"T": torch.log1p}, name="P")
    with torch.no_grad():
        g.wz.copy_(theta_w.to(DEV)); g.bias.copy_(theta_b.to(DEV))
    enc = MapEncoderPoissonNewton(g, lam=1.0, max_iter=30)
    Yd = Y.to(DEV)
    torch.manual_seed(seed)                       # fantasy noise stream (fresh each step)
    opt = torch.optim.SGD(g.parameters(), lr=LR)
    iters = []
    for k in range(K):
        opt.zero_grad()
        with torch.no_grad():
            z, _, _ = enc.sample(Yd)              # full data, detached (score-fn identity)
            yq = g.sample(z=g.sample_z(N_Q))      # fresh fantasy each step -> S-noise
            zq, _, _ = enc.sample(yq)
        m1 = g.zq_log(Yd, z=z).sum(-1).mean()
        m2 = g.zq_log(yq, z=zq).sum(-1).mean()
        loss = -(m1 - m2) + (L2 / N) * (g.wz ** 2).sum() / g.p / g.q
        loss.backward()
        opt.step()
        iters.append(g.wz.detach().cpu().clone())
    return torch.stack(iters)                     # (K, p, q)


def batch_means_se(iters, free, burn=BURN, n_batch=N_BATCH):
    """Batch-means SE of the Polyak average, per free functional. iters: (K, p, q)."""
    X = iters.reshape(iters.shape[0], -1)[:, free.reshape(-1)].numpy()   # (K, m)
    k0 = int(burn * X.shape[0])
    X = X[k0:]                                  # tail (stationary)
    Keff = X.shape[0]
    b = Keff // n_batch
    X = X[: b * n_batch]
    bm = X.reshape(n_batch, b, -1).mean(1)      # (n_batch, m) batch means
    grand = X.mean(0)
    # long-run variance sigma^2 (per coord), then Var(theta_bar) = sigma^2 / Keff
    sigma2 = b / (n_batch - 1) * ((bm - grand) ** 2).sum(0)
    return np.sqrt(sigma2 / Keff)               # (m,) SE of the Polyak average


def _stab_plot(dist, traj, that, k0):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(dist, lw=0.7); ax[0].axvline(k0, color="r", ls="--", lw=1, label="burn-in end")
    ax[0].set_xlabel("step k"); ax[0].set_ylabel(r"$\|\theta_k-\hat\theta\|$")
    ax[0].set_title("chain stability (should plateau)"); ax[0].legend()
    for j in range(min(5, traj.shape[1])):
        ax[1].plot(traj[:, j], lw=0.6)
        ax[1].axhline(that[j], color="k", ls=":", lw=0.6)
    ax[1].axvline(k0, color="r", ls="--", lw=1)
    ax[1].set_xlabel("step k"); ax[1].set_ylabel("loading value")
    ax[1].set_title(r"sample coords (dotted = $\hat\theta$)")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "polyak_stability.png"), dpi=130)
    print("saved polyak_stability.png")


def main():
    d = np.load(os.path.join(HERE, "results", "validation.npz"))
    Whats, Wboot, mask = d["Whats"], d["Wboot"], d["mask"].astype(bool)
    free = mask
    truth = varboot.make_truth(P, Q, wz_scale=WZ, seed=0, responses_per_latent=P)

    se_bm_all, se_boot_all = [], []
    for dd in range(N_TEST):
        t0 = time.time()
        Y = varboot.sample_data(truth, N, seed=1000 + dd)
        ft = varboot.fit_point(Y, Q, l2=L2 / N, device=DEV, seed=1000 + dd, wz_scale=WZ)
        tw, tb = ft.model.wz.detach().cpu(), ft.model.bias.detach().cpu()
        iters = polyak_chain(Y, tw, tb, seed=5000 + dd)

        # --- STABILITY CHECK: is the chain stationary around theta_hat? ---
        traj = iters.reshape(K, -1)[:, free.reshape(-1)].numpy()   # (K, m)
        that = tw.reshape(-1)[free.reshape(-1)].numpy()
        dist = np.linalg.norm(traj - that, axis=1)                 # ||theta_k - theta_hat||
        k0 = int(BURN * K)
        tail = slice(k0, K)
        kk = np.arange(k0, K)
        drift = np.polyfit(kk, dist[tail], 1)[0]                   # slope of ||.|| over tail (~0 if stationary)
        offset = np.linalg.norm(traj[tail].mean(0) - that) / (np.linalg.norm(that) + 1e-12)
        wig = dist[tail].mean()
        print(f"  [stab d{dd}] Polyak-mean offset/||W||={offset:.4f}  tail drift/step={drift:+.2e}  "
              f"mean||theta-hat||={wig:.4f}  finite={np.isfinite(dist).all()}", flush=True)
        if dd == 0:
            _stab_plot(dist, traj, that, k0)

        se_bm = batch_means_se(iters, free)
        se_boot = Wboot[dd][:, free].std(0, ddof=1) if Wboot[dd].ndim == 3 else \
                  Wboot[dd].reshape(Wboot[dd].shape[0], -1)[:, free.reshape(-1)].std(0, ddof=1)
        se_bm_all.append(se_bm); se_boot_all.append(se_boot)
        r = np.corrcoef(se_bm, se_boot)[0, 1]
        print(f"[{dd+1}/{N_TEST}] corr(SE_bm, SE_boot)={r:.3f}  "
              f"median SE_bm/SE_boot={np.median(se_bm/np.maximum(se_boot,1e-12)):.3f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    se_bm = np.concatenate(se_bm_all); se_boot = np.concatenate(se_boot_all)
    slope = np.sum(se_bm * se_boot) / np.sum(se_boot ** 2)   # through-origin slope
    r = np.corrcoef(se_bm, se_boot)[0, 1]
    print(f"\nPOOLED: corr={r:.3f}  through-origin slope (calibration c)={slope:.3f}")
    np.savez(os.path.join(HERE, "results", "polyak_bm.npz"),
             se_bm=se_bm, se_boot=se_boot, slope=slope, corr=r)

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        hi = max(se_bm.max(), se_boot.max()) * 1.05
        ax.plot([0, hi], [0, hi], "k--", lw=1, alpha=0.6, label="$y=x$")
        ax.plot([0, hi], [0, slope * hi], color="#d1495b", lw=1.2, alpha=0.8,
                label=f"fit slope $c={slope:.2f}$")
        ax.scatter(se_boot, se_bm, s=16, alpha=0.5, color="#2c6fbb")
        ax.set_xlabel("bootstrap SE (ground truth)"); ax.set_ylabel("batch-means SE (Polyak path)")
        ax.set_title(f"Batch-means vs bootstrap SE  (corr={r:.3f})")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(HERE, "polyak_bm_check.png"), dpi=140)
        print("saved polyak_bm_check.png")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
