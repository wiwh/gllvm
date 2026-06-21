"""Verify + benchmark block-diagonal exploitation for the GP-GLLVM (B=I).

  generating : z = L_Sigma eps           -> exact per-factor (Sigma block-diagonal)
  encoder    : solve A z = rhs,  A = Sigma^{-1} + (W^T W/s2) (x) I_K
               -> A is COUPLED across factors (W^T W), NOT block-diagonal when
                  the ell_k differ.  q-scalable route = prior-preconditioned CG
                  with a block-structured matvec (no (qK)^3 factorisation).

We check the fast paths reproduce the dense reference, then benchmark vs q.
"""
import time, numpy as np, torch
torch.set_default_dtype(torch.float64)

def vec(Z):      return Z.transpose(1, 2).reshape(Z.shape[0], -1)        # (n,K,q)->(n,qK) factor-major
def unvec(u, q, K): return u.reshape(u.shape[0], q, K).transpose(1, 2)   # (n,qK)->(n,K,q)

def blocks(coords, ells, jit=1e-4):                # per-factor kernels (n,q,K,K)
    d2 = ((coords[:, :, None, :] - coords[:, None, :, :]) ** 2).sum(-1)  # (n,K,K)
    Kq = torch.stack([torch.exp(-0.5 * d2 / e ** 2) for e in ells], 1)   # (n,q,K,K)
    return Kq + jit * torch.eye(coords.shape[1])

# ---------- DENSE reference (matches encode_b / sample_y_b) ----------
def Sigma_dense(coords, ells):
    Kq = blocks(coords, ells); n, q, K, _ = Kq.shape
    S = coords.new_zeros(n, q * K, q * K)
    for k in range(q): S[:, k*K:(k+1)*K, k*K:(k+1)*K] = Kq[:, k]
    return S

def gen_dense(eps, coords, ells):                  # eps (n,K,q)
    n, K, q = eps.shape
    L = torch.linalg.cholesky(Sigma_dense(coords, ells))
    return unvec((L @ vec(eps).unsqueeze(-1)).squeeze(-1), q, K)

def encode_dense(t, coords, ells, W, s2=1.0):      # t = (log1p(y)-b) (n,K,p)
    n, K, _ = t.shape; q = W.shape[1]
    L = torch.linalg.cholesky(Sigma_dense(coords, ells))
    Lit = torch.linalg.solve_triangular(L, torch.eye(q*K).expand(n, -1, -1), upper=False)
    A = Lit.transpose(1, 2) @ Lit + torch.kron(W.T @ W / s2, torch.eye(K))
    rhs = vec(t @ W / s2)
    return unvec(torch.cholesky_solve(rhs.unsqueeze(-1), torch.linalg.cholesky(A)).squeeze(-1), q, K)

# ---------- FAST block paths ----------
def gen_block(eps, coords, ells):                  # exact per-factor
    Lk = torch.linalg.cholesky(blocks(coords, ells))           # (n,q,K,K)
    zf = torch.einsum('nqij,nqj->nqi', Lk, eps.transpose(1, 2))
    return zf.transpose(1, 2)

def encode_cg(t, coords, ells, W, s2=1.0, iters=300, tol=1e-9):
    n, K, _ = t.shape; q = W.shape[1]
    Kq = blocks(coords, ells); Lk = torch.linalg.cholesky(Kq)  # (n,q,K,K)
    M = W.T @ W / s2
    rhs = vec(t @ W / s2)                                       # (n,qK)
    def Amv(v):                                                 # A @ v  (n,qK)
        vr = v.reshape(n, q, K)
        t1 = torch.cholesky_solve(vr.unsqueeze(-1), Lk).squeeze(-1)   # Sigma^{-1} v (block)
        t2 = torch.einsum('kl,nlt->nkt', M, vr)                       # (M (x) I_K) v
        return (t1 + t2).reshape(n, q * K)
    def Pinv(r):                                               # preconditioner ~ Sigma (block)
        return torch.einsum('nqij,nqj->nqi', Kq, r.reshape(n, q, K)).reshape(n, q * K)
    z = torch.zeros_like(rhs); r = rhs - Amv(z); zz = Pinv(r); p = zz.clone()
    rs = (r * zz).sum(-1, keepdim=True)
    it = 0
    for it in range(1, iters + 1):
        Ap = Amv(p); alpha = rs / ((p * Ap).sum(-1, keepdim=True) + 1e-300)
        z = z + alpha * p; r = r - alpha * Ap
        if r.norm(dim=-1).max() < tol: break
        zz = Pinv(r); rs_new = (r * zz).sum(-1, keepdim=True)
        p = zz + (rs_new / (rs + 1e-300)) * p; rs = rs_new
    return unvec(z, q, K), it

# ================= verify =================
torch.manual_seed(0)
n, K, q, p = 6, 40, 4, 30
coords = torch.sort(torch.rand(n, K) * K, 1).values[..., None]     # 1-D coords (n,K,1)
ells = torch.tensor([0.5, 1.5, 4.0, 9.0])[:q]
W = torch.randn(p, q) * 0.5
eps = torch.randn(n, K, q); t = torch.randn(n, K, p)

zg_d, zg_b = gen_dense(eps, coords, ells), gen_block(eps, coords, ells)
ze_d = encode_dense(t, coords, ells, W)
ze_c, iters = encode_cg(t, coords, ells, W)
print(f"generating  : max|dense-block| = {(zg_d-zg_b).abs().max():.2e}")
print(f"encoder      : max|dense-CG|    = {(ze_d-ze_c).abs().max():.2e}   (CG iters={iters})")

# ================= benchmark vs q =================
print("\nbenchmark encode (n=32, K=60), dense (qK)^3 vs CG block:")
print(f"{'q':>4} {'qK':>5} {'dense_s':>9} {'cg_s':>8} {'cg_iters':>9} {'speedup':>8} {'max_err':>9}")
n, K = 32, 60
for q in [2, 4, 8, 16, 32]:
    torch.manual_seed(q)
    coords = torch.sort(torch.rand(n, K) * K, 1).values[..., None]
    ells = torch.linspace(0.5, 10.0, q)
    W = torch.randn(p, q) * 0.5; t = torch.randn(n, K, p)
    t0 = time.perf_counter(); zd = encode_dense(t, coords, ells, W); td = time.perf_counter() - t0
    t0 = time.perf_counter(); zc, it = encode_cg(t, coords, ells, W); tc = time.perf_counter() - t0
    print(f"{q:>4} {q*K:>5} {td:>9.3f} {tc:>8.3f} {it:>9} {td/tc:>7.1f}x {(zd-zc).abs().max():>9.1e}")
