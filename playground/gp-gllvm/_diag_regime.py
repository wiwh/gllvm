"""Panel-2 isolation (cheap, no full-block fits): does the GP-GLLVM with FITTED ℓ
beat the independent model across sparsity? Moderate smoothness ℓ=[2,8]."""
import sys, numpy as np, torch
sys.path.insert(0, "/home/willwhite/GitHub/gllvm/src")
exec(open("/home/willwhite/GitHub/gllvm/playground/gp-gllvm/_subsample_demo.py").read().split("NSEED, BG")[0])  # reuse core+fit

ELL_TRUE[:] = [2.0, 8.0]
NSEED, BG, G, K = 8, 16, 200, 25
offs = matched_offsets(K)
print(f"GP data ℓ_true={ELL_TRUE}, window K={K}, {NSEED} seeds, fitted ℓ")
print(f"{'b0':>5} {'meanc':>6} | {'GP(fit ℓ)':>10} {'indep':>8}  gap%  | {'ℓ̂':>14}")
for b0 in [-2.5, -1.5, -0.5, 0.5]:
    mc, pg, pi, eh = [], [], [], []
    for s in range(NSEED):
        Yf, Wt, _ = gen_data(BG=BG, G=G, b0=b0, seed=s)
        mc.append(Yf.double().mean().item())
        smp = lambda: sub_matched(Yf, offs, 96, DT)
        Wg, e = fit(Yf, smp, fit_ell=True, ell_init=(1.5, 5.0), steps=700, seed=s)
        pg.append(procrustes_error(Wt, Wg)); eh.append(e)
        pi.append(procrustes_error(Wt, fit(Yf, smp, fit_ell=False, ell_fixed=(1e-3, 1e-3), steps=700, seed=s)[0]))
    mc, pg, pi = np.mean(mc), np.mean(pg), np.mean(pi); eh = np.mean(eh, 0)
    print(f"{b0:>5.1f} {mc:>6.2f} | {pg:>10.3f} {pi:>8.3f}  {100*(pi-pg)/pi:>4.0f}  | [{eh[0]:.2f},{eh[1]:.2f}]")
