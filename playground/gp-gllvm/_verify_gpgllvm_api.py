"""Verify the new gllvm.GPGLLVM / GPZQEFitter API: recover per-factor length-scales
and loadings on synthetic GP-GLLVM data (Poisson, B=I, distinct ell)."""
import sys, torch, numpy as np
sys.path.insert(0, "/home/willwhite/GitHub/gllvm/src")
from gllvm import GPGLLVM, GPZQEFitter, PoissonGLM, procrustes_error
torch.set_default_dtype(torch.float64)

Q, P, B, G, DT = 2, 30, 80, 40, 1.0
ELL_TRUE = [1.0, 3.0]

def make_true(seed):
    torch.manual_seed(seed)
    g = GPGLLVM(Q, P, input_dim=1, lengthscale=ELL_TRUE)
    g.add_glm(PoissonGLM, idx=range(P), params={"T": torch.log1p}, name="P")
    with torch.no_grad():
        g.wz.normal_(0, 0.7); g.bias.fill_(0.5)
    return g

def make_data(g_true, seed):
    torch.manual_seed(1000 + seed)
    t = (torch.arange(G) * DT).double()
    coords = t[None, :, None].expand(B, G, 1)                 # (B,G,1)
    y = g_true.sample(coords)                                  # (B,G,P)
    n = B * G
    return (y.reshape(n, P),
            coords.reshape(n, 1),
            torch.arange(B).repeat_interleave(G))             # group ids

print(f"GP-GLLVM API verify: q={Q} p={P} groups={B} G={G} ell_true={ELL_TRUE}")
print(f"{'seed':>4} {'ell_hat':>16} {'procW':>8} {'mean_count':>10}")
rows = []
for s in range(5):
    gt = make_true(s)
    y, coords, groups = make_data(gt, s)
    fit = GPGLLVM(Q, P, input_dim=1, lengthscale=2.0)         # fresh, ell init away from truth
    fit.add_glm(PoissonGLM, idx=range(P), params={"T": torch.log1p}, name="P")
    with torch.no_grad():
        fit.wz.normal_(0, 0.7)
    ft = GPZQEFitter(fit, K=G, steps=1200, batch=128, warmup=300, lr=0.03,
                     device="cpu", seed=s).fit(y, coords, groups=groups)
    ell = sorted(ft.lengthscales_.tolist())
    pw = procrustes_error(gt.wz.detach(), fit.wz.detach())
    rows.append((ell[0], ell[1], pw))
    print(f"{s:>4} [{ell[0]:.2f}, {ell[1]:.2f}]{'':>6} {pw:>8.3f} {y.double().mean():>10.2f}")

a = np.array(rows)
print(f"mean [{a[:,0].mean():.2f}, {a[:,1].mean():.2f}]  procW {a[:,2].mean():.3f}  (true {ELL_TRUE})")
