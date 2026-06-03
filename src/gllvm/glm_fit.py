"""Batched GLM fitting helpers used by notebooks.

Functions:
 - initial_gaussian_fit(X, Y): least-squares on T(Y)=log1p(Y), returns B0 (q x n)
 - poisson_newton_batch(X, Y, B0, ...): batched Newton updates for Poisson GLM

All arrays expected as PyTorch tensors with shapes:
 - X: (p, q)
 - Y: (p, n)
 - B: (q, n)

Implemented to be numerically stable and batch the qxq solves over n columns.
"""

from typing import Tuple, Dict, Optional
import torch


def initial_gaussian_fit(X: torch.Tensor, Y: torch.Tensor, offset: Optional[torch.Tensor] = None, eps: float = 1e-6) -> torch.Tensor:
    """Return least-squares fit B0 solving X B ≈ log1p(Y) - offset for each column of Y.

    Args:
        X: (p, q) design matrix
        Y: (p, n) integer counts
        offset: optional (p, n) offset added to the linear predictor.
        eps: ridge fallback regulariser, used only when lstsq fails.

    Returns:
        B0: (q, n)
    """
    T = torch.log1p(Y)
    if offset is not None:
        if offset.shape != Y.shape:
            raise ValueError(f"offset shape {offset.shape} != Y shape {Y.shape}")
        T = T - offset
    # On CUDA only "gels" (QR) is available; on CPU use "gelsd" (SVD, rank-safe).
    # Fall back to ridge normal equations if lstsq raises (e.g. rank-deficient W).
    driver = "gels" if X.is_cuda else "gelsd"
    try:
        return torch.linalg.lstsq(X, T, driver=driver).solution
    except Exception:
        # Ridge fallback: (X^T X + eps I)^{-1} X^T T
        XT  = X.T
        XtX = XT @ X
        q   = X.shape[1]
        reg = XtX + eps * torch.eye(q, device=X.device, dtype=X.dtype)
        try:
            return torch.linalg.solve(reg, XT @ T)
        except Exception:
            return torch.zeros(X.shape[1], T.shape[1], device=X.device, dtype=X.dtype)


def poisson_newton_batch(
    X: torch.Tensor,
    Y: torch.Tensor,
    B0: torch.Tensor,
    offset: Optional[torch.Tensor] = None,
    lam: float = 1,
    max_iter: int = 50,
    tol: float = 1e-6,
    damp: float = 1.0,
    verbose: bool = False,
    max_halvings: int = 10,
) -> Tuple[torch.Tensor, Dict]:
    """Batched Newton updates for Poisson GLM (log link).

    Solves for B (q, n) minimising negative Poisson log-likelihood plus
    (lam/2)*||B||_F^2, starting from B0.

    The update for each column i uses the Hessian H_i = X^T diag(mu_i) X + lam I
    and gradient g_i = X^T (mu_i - y_i) + lam * b_i.  We solve these qxq
    systems in a batched fashion using `torch.linalg.solve`.

    Returns final B and diagnostics dict.
    """
    p, q = X.shape
    _, n = Y.shape
    B = B0.clone().to(X.device)
    eye = torch.eye(q, device=X.device, dtype=X.dtype)
    it = 0
    rel_change = 0.0
    total_halvings = 0
    boundary = False
    for it in range(1, max_iter + 1):
        eta = X @ B                      # (p, n)
        if offset is not None:
            if offset.shape != eta.shape:
                raise ValueError(f"offset shape {offset.shape} != eta shape {eta.shape}")
            eta = eta + offset
        mu = torch.exp(eta)             # (p, n)

        # approximate negative log-likelihood (up to additive constant):
        # nll = sum(mu - y * log(mu)). Use small eps for log stability.
        nll_old = torch.sum(mu - Y * eta) + 0.5 * lam * torch.sum(B * B)

        # gradient: q x n
        g = X.T @ (mu - Y) + lam * B

        # Hessian batch: for each column j, H_j[a,b] = sum_p X[p,a] * mu[p,j] * X[p,b]
        # produce H as (q, q, n) via einsum
        H = torch.einsum('pa,pj,pb->abj', X, mu, X)  # (q, q, n)
        H = H + (lam + 1e-8) * eye[:, :, None]

        # reshape for batch solve: (n, q, q) and rhs (n, q, 1)
        H_b = H.permute(2, 0, 1).contiguous()      # (n, q, q)
        g_b = g.T.unsqueeze(-1).contiguous()       # (n, q, 1)

        # solve H_b @ delta_b = g_b  => delta_b (n, q, 1)
        try:
            delta_b = torch.linalg.solve(H_b, g_b)
        except RuntimeError:
            # fallback to looped solve if batch solve fails for numerical reasons
            delta = torch.empty_like(g)
            for j in range(n):
                Hj = H[:, :, j]
                gj = g[:, j:j+1]
                dj = torch.linalg.solve(Hj, gj)
                delta[:, j] = dj.squeeze(-1)
            delta = delta
        else:
            delta = delta_b.squeeze(-1).T  # (q, n)

        B_old = B
        B_new = B - damp * delta

        # Evaluate new objective and perform step-halving if it increases or is non-finite
        eta_new = X @ B_new
        if offset is not None:
            eta_new = eta_new + offset
        mu_new = torch.exp(eta_new)
        nll_new = torch.sum(mu_new - Y * eta_new) + 0.5 * lam * torch.sum(B_new * B_new)

        halvings = 0
        # require a (small) decrease in nll to accept the step
        nll_tol = 1e-9
        while (not torch.isfinite(nll_new)) or (nll_new > nll_old + nll_tol):
            if halvings >= max_halvings:
                # give up on this update: revert to old B and mark boundary
                B_new = B_old
                boundary = True
                break
            # step-halving
            B_new = 0.5 * (B_new + B_old)
            eta_new = X @ B_new
            if offset is not None:
                eta_new = eta_new + offset
            mu_new = torch.exp(eta_new)
            nll_new = torch.sum(mu_new - Y * eta_new) + 0.5 * lam * torch.sum(B_new * B_new)
            halvings += 1

        total_halvings += halvings

        rel_change = torch.norm(B_new - B_old) / (torch.norm(B_old) + 1e-12)
        B = B_new
        if verbose:
            print(f"Newton it={it:3d} rel_change={rel_change:.3e} halvings={halvings}")
        if rel_change < tol or boundary:
            break

    diagnostics = {"iters": it, "rel_change": float(rel_change)}
    return B, diagnostics


if __name__ == "__main__":
    # small smoke test
    torch.manual_seed(0)
    p, q, n = 50, 5, 10
    X = torch.randn(p, q)
    B_true = 0.5 * torch.randn(q, n)
    # no-offset case
    eta = X @ B_true
    Y = torch.poisson(torch.exp(eta))
    B0 = initial_gaussian_fit(X, Y)
    B_hat, info = poisson_newton_batch(X, Y, B0, lam=1e-3, max_iter=50)
    print("smoke no-offset: rel_err=", torch.norm(B_hat - B_true) / torch.norm(B_true))

    # with-offset case
    offset_true = 0.2 * torch.randn(p, n)
    eta2 = X @ B_true + offset_true
    Y2 = torch.poisson(torch.exp(eta2))
    B0_off = initial_gaussian_fit(X, Y2, offset=offset_true)
    B_hat2, info2 = poisson_newton_batch(X, Y2, B0_off, offset=offset_true, lam=1e-3, max_iter=50)
    print("smoke with-offset: rel_err=", torch.norm(B_hat2 - B_true) / torch.norm(B_true))
