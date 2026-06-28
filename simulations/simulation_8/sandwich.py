"""
Full-matrix sandwich SE for ZQE, via the GODAMBE identity on JOINT (prior-sampled) draws.

Sandwich:  Cov(theta_hat) = A^-1 S A^-T / n.
  ham   S = Cov_data( g_i ),         g_i = T(y_i) * dEta/dtheta  (the per-sample data gradient)
  bread A = -E[d psi/d theta] = E[ psi * s^T ]   (Godambe sensitivity identity; holds because the
            ZQE centering makes E_{f_theta}[psi]=0). Evaluated on JOINT fantasy draws
            (z_prior ~ p(z), Y_q ~ Poisson|z_prior):  A = Cov( c , s ),
            c = T(Y_q)*dEta/dtheta  (centering integrand),
            s = (Y_q - lambda_q)*dEta/dtheta  (COMPLETE-DATA Poisson score; z_prior is theta-free,
                so the joint score collapses to the analytic conditional score -- NO marginal score).

Validation: SE on the lower-tri free loadings, compared to the CRN bootstrap SE in
results/validation.npz. Success = SE_sandwich ~ SE_boot (slope~1, corr~1). The detached-encoder
consistency is the assumption being tested.

Run: python sandwich.py
"""
import os, sys, time
sys.path.insert(0, os.path.abspath("../../src"))
import numpy as np
import torch
import varboot
from gllvm.gllvm_module import GLLVM
from gllvm.glms import PoissonGLM
from gllvm.encoder import MapEncoderPoissonNewton

P, Q, N = 50, 2, 500
L2, WZ = 0.001, 0.5
DEV = "cuda" if torch.cuda.is_available() else "cpu"
HERE = os.path.dirname(os.path.abspath(__file__))
N_TEST = 6
M_FANT = 40000          # fantasy draws to estimate the d x d bread
CLAMP = 10.0            # matches PoissonGLM linpar clamp


def model_at(tw, tb):
    g = GLLVM(latent_dim=Q, output_dim=P, bias=True, lower_tri=True).to(DEV)
    g.add_glm(PoissonGLM, idx=list(range(P)), params={"T": torch.log1p}, name="P")
    with torch.no_grad():
        g.wz.copy_(tw.to(DEV)); g.bias.copy_(tb.to(DEV))
    return g


def grad_vec(Tval, zfeat, mask):
    """Structured dEta/dtheta * Tval. Tval:(B,p), zfeat:(B,q), mask:(p,q) bool.
    Returns (B, n_wfree + p): [ W-block outer(Tval,zfeat)[mask] , b-block Tval ]."""
    Wpart = Tval[:, :, None] * zfeat[:, None, :]      # (B, p, q)
    Wfree = Wpart[:, mask]                             # (B, n_wfree)
    return torch.cat([Wfree, Tval], dim=1)            # (B, d)


def _g_and_score(enc, W, b, Y, mask, I):
    """Return (g, s_marginal) for observations Y: g = log1p(Y)*[zhat,1] (ham gradient);
    s_marginal = CLOSED-FORM Laplace posterior score via Sigma = (W^T diag(lam) W + I)^-1 at zhat."""
    Yd = Y.double()
    z, _, _ = enc.sample(Y)
    Z = z.detach().double()                                        # (m,q) MAP
    g = grad_vec(torch.log1p(Yd), Z, mask)                         # (m,d)
    linpar = (Z @ W.T + b).clamp(max=CLAMP)                        # (m,p)
    lam = linpar.exp()
    H = torch.einsum("pq,mp,pr->mqr", W, lam, W) + I              # (m,q,q)
    Sig = torch.linalg.inv(H)
    quad = torch.einsum("pq,mqr,pr->mp", W, Sig, W)               # (m,p) w_j^T Sig w_j
    lbar = (linpar + 0.5 * quad).clamp(max=CLAMP).exp()          # (m,p) E[lam_j(u)]
    Sigw = torch.einsum("mqr,pr->mpq", Sig, W)                    # (m,p,q)
    zc = Z[:, None, :] + Sigw
    Wblock = torch.einsum("mp,mq->mpq", Yd, Z) - lbar[:, :, None] * zc
    bblock = Yd - lbar
    s = torch.cat([Wblock[:, mask], bblock], dim=1)               # (m,d)
    return g, s


@torch.no_grad()
def sandwich_se(g, Y, mask, m_fant=M_FANT, seed=0):
    """Sandwich with the LAPLACE (encoder-posterior) marginal score, post-estimation.
    ham   S = Cov_data(g_i)              (n data points)
    bread A = Cov_fantasy(g_q, s_q)      (M fantasy draws -> well-conditioned d x d)
    Cov(theta_hat) = A^-1 S A^-T / n."""
    enc = MapEncoderPoissonNewton(g, lam=1.0, max_iter=30)
    W = g.wz.detach().double(); b = g.bias.detach().double()
    I = torch.eye(Q, dtype=torch.double, device=DEV)

    # ham over data
    g_data, _ = _g_and_score(enc, W, b, Y, mask, I)
    S = torch.from_numpy(np.cov(g_data.cpu().numpy().T)).to(DEV).double()

    # bread over many fantasy draws (M >> d), low-variance Laplace score
    gen = torch.Generator(device=DEV).manual_seed(seed)
    zpri = torch.randn(m_fant, Q, generator=gen, device=DEV, dtype=torch.double)
    Yq = torch.poisson((zpri @ W.T + b).clamp(max=CLAMP).exp(), generator=gen)
    gq, sq = _g_and_score(enc, W, b, Yq, mask, I)
    gc = gq - gq.mean(0, keepdim=True); sc = sq - sq.mean(0, keepdim=True)
    A = (gc.T @ sc) / (m_fant - 1)                                # (d,d), well-sampled

    Ainv = torch.linalg.inv(A)
    Cov = Ainv @ S @ Ainv.T / N
    n_wfree = int(mask.sum())
    se = torch.sqrt(torch.clamp(torch.diag(Cov)[:n_wfree], min=0)).cpu().numpy()
    return se, float(torch.linalg.cond(A))


def main():
    d = np.load(os.path.join(HERE, "results", "validation.npz"))
    Wboot, mask_np = d["Wboot"], d["mask"].astype(bool)
    mask = torch.from_numpy(mask_np).to(DEV)
    truth = varboot.make_truth(P, Q, wz_scale=WZ, seed=0, responses_per_latent=P)

    se_s_all, se_b_all = [], []
    for dd in range(N_TEST):
        t0 = time.time()
        Y = varboot.sample_data(truth, N, seed=1000 + dd).to(DEV)
        ft = varboot.fit_point(Y, Q, l2=L2 / N, device=DEV, seed=1000 + dd, wz_scale=WZ)
        se_s, condA = sandwich_se(ft.model, Y, mask, seed=4242 + dd)
        se_b = Wboot[dd][:, mask_np].std(0, ddof=1)
        se_s_all.append(se_s); se_b_all.append(se_b)
        r = np.corrcoef(se_s, se_b)[0, 1]
        rat = np.median(se_s / np.maximum(se_b, 1e-12))
        print(f"[{dd+1}/{N_TEST}] corr={r:.3f}  median SE_sand/SE_boot={rat:.3f}  "
              f"cond(A)={condA:.1e}  ({time.time()-t0:.0f}s)", flush=True)

    se_s = np.concatenate(se_s_all); se_b = np.concatenate(se_b_all)
    slope = np.sum(se_s * se_b) / np.sum(se_b ** 2)
    r = np.corrcoef(se_s, se_b)[0, 1]
    print(f"\nPOOLED: corr={r:.3f}  through-origin slope={slope:.3f}  "
          f"(target: corr~1, slope~1)")
    np.savez(os.path.join(HERE, "results", "sandwich.npz"), se_s=se_s, se_b=se_b, slope=slope, corr=r)

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        hi = max(se_s.max(), se_b.max()) * 1.05
        ax.plot([0, hi], [0, hi], "k--", lw=1, label="$y=x$")
        ax.scatter(se_b, se_s, s=16, alpha=0.5, color="#2c6fbb")
        ax.set_xlabel("bootstrap SE (truth)"); ax.set_ylabel("sandwich SE (Godambe)")
        ax.set_title(f"Sandwich vs bootstrap  (corr={r:.3f}, slope={slope:.2f})")
        ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
        fig.savefig(os.path.join(HERE, "sandwich_check.png"), dpi=140)
        print("saved sandwich_check.png")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
