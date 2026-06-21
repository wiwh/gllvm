import torch
from torch import nn
from gllvm.glm_fit import initial_gaussian_fit, poisson_newton_batch, bernoulli_newton_batch


class Encoder(nn.Module):
    def __init__(self, input_dim=5, latent_dim=1, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)

    def forward(self, y):
        h = self.net(y)
        mu = self.mean(h)
        logvar = self.logvar(h)
        return mu, logvar

    def sample(self, y):
        mu, logvar = self.forward(y)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

    # ----------------------------------------------------------
    # ELBO loss (encoder-only)
    # ----------------------------------------------------------
    def loss(self, y, gllvm):
        z, mu, logvar = self.sample(y)

        # log p(y|z)
        logpy = gllvm.log_prob(y, z=z).sum(dim=-1)

        # KL(q||p)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

        elbo = (logpy - kl).mean()
        return -elbo, elbo.item()  # loss, scalar ELBO


class EncoderPosteriorSupervised(nn.Module):
    """
    Supervised posterior encoder.
    Learns q(z|y) = N(mu(y), sigma^2(y)) using synthetic (z_true, y)
    generated from GLLVM.

    Loss = Gaussian NLL of z_true under q(z|y):
       0.5 * ((z - mu)^2 / exp(logvar)) + 0.5 * logvar
    """

    def __init__(self, input_dim=5, latent_dim=1, hidden=32):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        self.mean = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)

    def forward(self, y):
        h = self.net(y)
        mu = self.mean(h)
        logvar = self.logvar(h)
        return mu, logvar

    # sample from learned posterior
    def sample(self, y):
        mu, logvar = self.forward(y)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

    # ----------------------------------------------------------
    # Supervised posterior learning:
    # Sample (z_true, y) from the GLLVM, and fit q(z|y)
    # to the true z.
    # ----------------------------------------------------------
    def loss(self, _, gllvm, batch_size=512):
        with torch.no_grad():
            z_true = gllvm.sample_z(batch_size)
            y = gllvm.sample(z=z_true)

        mu, logvar = self.forward(y)

        # Gaussian NLL: -log q(z_true | y)
        inv_sigma2 = torch.exp(-logvar)
        nll = 0.5 * ((z_true - mu) ** 2 * inv_sigma2 + logvar).mean()

        # some regularization o z:
        regul = 0.5 * (mu**2 + torch.exp(logvar)).mean()
        nll += regul

        return nll, -nll.item()


class EncoderMAP(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
        )
        self.mu = nn.Linear(hidden, latent_dim)

    def forward(self, y):
        return self.mu(self.net(y))

    def sample(self, y):
        z = self.forward(y)
        return z, z, torch.zeros_like(z)

    def loss(self, y, gllvm):
        z = self.forward(y)

        # prior log p(z)
        lpz = -0.5 * (z**2).sum(dim=-1)

        # decoder likelihood log p(y|z)
        linpar = gllvm.forward(z)
        logp_ygz = gllvm.log_prob(y, linpar=linpar).sum(dim=-1)

        # MAP = maximize (log p(y|z) + log p(z))
        loss = -(logp_ygz + lpz).mean()
        return loss, -loss.item()


class EncoderMAPSupervised(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, latent_dim)

    def forward(self, y):
        return self.mu(self.net(y))

    def sample(self, y):
        z = self.forward(y)
        return z, z, torch.zeros_like(z)

    def loss(self, y, gllvm):
        with torch.no_grad():
            z_sim = gllvm.sample_z(len(y))
            y_sim = gllvm.sample(z=z_sim)

        z_est = self.forward(y_sim)
        # prior log p(z)

        loss = ((z_sim - z_est) ** 2).sum(dim=-1).mean()
        loss_regul = 0.5 * (z_est**2).mean()
        loss += loss_regul

        return loss, -loss.item()


class MapEncoderGaussianLog1p(nn.Module):
    """
    Parameter-free analytical MAP encoder.

    Assumes a Gaussian proxy model in log1p-space:

        log1p(y) | z  ~  N(W z + b,  sigma^2 I)
        z             ~  N(0, I)

    The MAP is the closed-form ridge solution:

        z_MAP(y; theta) = (sigma^2 I + W^T W)^{-1} W^T (log1p(y) - b)

    where W = gllvm.wz  (p x q)  and  b = gllvm.bias  (p,).

    Key properties
    --------------
    * No learnable parameters — nothing to train, no warmup.
    * Depends *explicitly* on the current decoder theta = (W, b): the MAP
      improves automatically as the decoder is updated.
    * The q x q solve costs O(q^3), negligible for q << p.
    * Fully differentiable: gradient flows through W and b if called
      outside torch.no_grad().
    * sigma2 acts as a ridge penalty / prior precision ratio; sigma2=1
      corresponds to the standard N(0,I) prior matching the GLLVM's latent prior.
    """

    def __init__(self, gllvm, sigma2: float = 1.0):
        super().__init__()
        self.gllvm = gllvm   # live reference — always uses current W, b
        self.sigma2 = sigma2

    def forward(self, y):
        W = self.gllvm.wz           # (p, q)
        b = (self.gllvm.bias
             if self.gllvm.bias is not None
             else torch.zeros(W.shape[0], device=W.device, dtype=W.dtype))

        t_y = torch.log1p(y.float())            # (n, p)
        rhs = (t_y - b.unsqueeze(0)) @ W        # (n, q)  =  (log1p(y)-b)^T W

        A = (self.sigma2 * torch.eye(W.shape[1], device=W.device, dtype=W.dtype)
             + W.T @ W)                          # (q, q)

        # solve A z^T = rhs^T  ->  z_MAP = (A^{-1} rhs^T)^T
        z_map = torch.linalg.solve(A, rhs.T).T  # (n, q)
        return z_map

    def sample(self, y):
        """Drop-in for Encoder.sample() — deterministic delta-mass surrogate."""
        z = self.forward(y)
        return z, z, torch.full_like(z, float("-inf"))

    def loss(self, y, gllvm=None, **kwargs):
        """No parameters to optimise — return a zero loss."""
        dummy = next(self.gllvm.parameters())
        return torch.zeros(1, device=dummy.device, requires_grad=True), 0.0


class MapEncoderPoissonNewton(nn.Module):
    """
    Parameter-free exact Poisson MAP encoder via batched Newton GLM.

    Solves, for each observation y_i:

        z* = argmax_z { sum_j [y_ij * eta_j(z) - exp(eta_j(z))] - 0.5 ||z||^2 }

    where eta_j(z) = w_j^T z + b_j, using the fast vectorised Newton solver
    from ``glm_fit.py`` with a Gaussian log1p warm-start.

    Parameters
    ----------
    gllvm       : GLLVM  — live reference; always uses current W, b.
    lam         : float  — prior precision (matches N(0,I) at lam=1).
    max_iter    : int    — Newton iterations.
    max_halvings: int    — step-halving budget per iteration.
    tol         : float  — convergence tolerance on ||Δz||.
    """

    def __init__(self, gllvm, lam: float = 1.0, max_iter: int = 30,
                 max_halvings: int = 10, tol: float = 1e-6):
        super().__init__()
        self.gllvm        = gllvm
        self.lam          = lam
        self.max_iter     = max_iter
        self.max_halvings = max_halvings
        self.tol          = tol

    def forward(self, y):
        W = self.gllvm.wz.detach()          # (p, q)
        b = (self.gllvm.bias.detach()
             if self.gllvm.bias is not None
             else torch.zeros(W.shape[0], device=W.device, dtype=W.dtype))  # (p,)

        Y_t    = y.float().T.contiguous()              # (p, batch)
        offset = b.unsqueeze(1).expand_as(Y_t)         # (p, batch)

        with torch.no_grad():
            B0 = initial_gaussian_fit(W, Y_t, offset=offset)          # (q, batch)
            B_hat, _ = poisson_newton_batch(
                X=W, Y=Y_t, B0=B0, offset=offset,
                lam=self.lam, max_iter=self.max_iter,
                max_halvings=self.max_halvings, tol=self.tol, verbose=False,
            )
        return B_hat.T.contiguous()   # (batch, q)

    def sample(self, y):
        """Drop-in for Encoder.sample() — deterministic delta-mass surrogate."""
        z = self.forward(y)
        return z, z, torch.full_like(z, float("-inf"))

    def loss(self, y, gllvm=None, **kwargs):
        """No parameters to optimise — return a zero loss."""
        dummy = next(self.gllvm.parameters())
        return torch.zeros(1, device=dummy.device, requires_grad=True), 0.0


class MapEncoderBernoulliNewton(nn.Module):
    """
    Parameter-free exact Bernoulli MAP encoder via batched Newton/IRLS.

    Solves, for each observation y_i (binary):

        z* = argmax_z { sum_j [y_ij * eta_j(z) - softplus(eta_j(z))] - (lam/2) ||z||^2 }

    where eta_j(z) = w_j^T z + b_j (logit link), using
    ``glm_fit.bernoulli_newton_batch``.  This is the binary analogue of
    ``MapEncoderPoissonNewton``: the proper logit posterior mode, far better
    conditioned than the ``±2`` Gaussian-proxy at small n.

    Parameters
    ----------
    gllvm    : GLLVM  — live reference; always uses the current W, b.
    lam      : float  — prior precision (matches N(0, I) at lam=1).
    max_iter : int    — Newton iterations.
    tol      : float  — convergence tolerance on relative ||Δz||.
    """

    def __init__(self, gllvm, lam: float = 1.0, max_iter: int = 30,
                 max_halvings: int = 10, tol: float = 1e-6):
        super().__init__()
        self.gllvm = gllvm
        self.lam = lam
        self.max_iter = max_iter
        self.max_halvings = max_halvings
        self.tol = tol

    def forward(self, y):
        W = self.gllvm.wz.detach()                       # (p, q)
        b = (self.gllvm.bias.detach()
             if self.gllvm.bias is not None
             else torch.zeros(W.shape[0], device=W.device, dtype=W.dtype))

        Y_t = y.float().T.contiguous()                   # (p, batch)
        offset = b.unsqueeze(1).expand_as(Y_t)           # (p, batch)
        B0 = torch.zeros(W.shape[1], Y_t.shape[1], device=W.device, dtype=W.dtype)

        with torch.no_grad():
            B_hat, _ = bernoulli_newton_batch(
                X=W, Y=Y_t, B0=B0, offset=offset,
                lam=self.lam, max_iter=self.max_iter,
                max_halvings=self.max_halvings, tol=self.tol, verbose=False,
            )
        return B_hat.T.contiguous()                       # (batch, q)

    def sample(self, y):
        """Drop-in for Encoder.sample() — deterministic delta-mass surrogate."""
        z = self.forward(y)
        return z, z, torch.full_like(z, float("-inf"))

    def loss(self, y, gllvm=None, **kwargs):
        """No parameters to optimise — return a zero loss."""
        dummy = next(self.gllvm.parameters())
        return torch.zeros(1, device=dummy.device, requires_grad=True), 0.0


class GaussianPosteriorEncoderLog1p(nn.Module):
    """
    Parameter-free encoder that draws exact samples from the Gaussian posterior.

    Proxy model in log1p-space (misspecified but tractable):

        log1p(y) | z  ~  N(W z + b,  sigma^2 I)
        z             ~  N(0, I)

    Exact posterior:

        q(z | y)  =  N(mu, Sigma)
        Sigma     =  sigma^2 * (W^T W + sigma^2 I)^{-1}       (q x q)
        mu        =  Sigma / sigma^2 * W^T (log1p(y) - b)
                  =  (W^T W + sigma^2 I)^{-1} W^T (log1p(y) - b)

    The mean mu is identical to the MAP solution; the novelty is drawing
    samples from the full posterior, not a point-mass.  This is equivalent
    to non-amortised variational inference under the misspecified Gaussian
    model — but because the ZQE estimating equation satisfies the score-function
    identity, the encoder misspecification does *not* bias the decoder updates.

    The posterior covariance is shared across observations (it only depends on
    W, not on y), so we compute its Cholesky factor once per call.
    """

    def __init__(self, gllvm, sigma2: float = 1.0):
        super().__init__()
        self.gllvm = gllvm
        self.sigma2 = sigma2

    def _posterior_params(self, y):
        """Return (mu, L) where L is the lower Cholesky of Sigma."""
        W = self.gllvm.wz           # (p, q)
        b = (self.gllvm.bias
             if self.gllvm.bias is not None
             else torch.zeros(W.shape[0], device=W.device, dtype=W.dtype))

        t_y = torch.log1p(y.float())            # (n, p)
        rhs = (t_y - b.unsqueeze(0)) @ W        # (n, q)

        A = (self.sigma2 * torch.eye(W.shape[1], device=W.device, dtype=W.dtype)
             + W.T @ W)                          # (q, q)

        # mu = A^{-1} rhs^T  transposed back to (n, q)
        mu = torch.linalg.solve(A, rhs.T).T      # (n, q)

        # Sigma = sigma^2 * A^{-1}  — Cholesky of Sigma
        # L L^T = sigma^2 A^{-1}   =>  L = sqrt(sigma^2) * chol(A^{-1})
        #                              = sqrt(sigma^2) * solve(chol(A), I)^T
        L_A = torch.linalg.cholesky(A)           # (q, q), lower
        I_q = torch.eye(W.shape[1], device=W.device, dtype=W.dtype)
        # L_A L_A^T = A  =>  A^{-1} = (L_A^{-1})^T L_A^{-1}
        # Cholesky of Sigma = sqrt(sigma2) * L_A^{-T}  (upper-triangular of A^{-1})
        L_Ainv = torch.linalg.solve_triangular(L_A, I_q, upper=False)  # (q, q)
        L_Sigma = (self.sigma2 ** 0.5) * L_Ainv.T   # (q, q), upper; use as scale

        return mu, L_Sigma

    def forward(self, y):
        """Return a sample z ~ q(z|y) via the reparameterisation trick."""
        mu, L_Sigma = self._posterior_params(y)
        eps = torch.randn_like(mu)               # (n, q)
        # L_Sigma is upper-triangular: z = mu + eps @ L_Sigma  (each row scaled)
        z = mu + eps @ L_Sigma                   # (n, q)
        return z

    def sample(self, y):
        """Drop-in for Encoder.sample() — returns (z_sample, mu, log_std)."""
        mu, L_Sigma = self._posterior_params(y)
        eps = torch.randn_like(mu)
        z = mu + eps @ L_Sigma
        # log_std: diagonal of L_Sigma (approximate scalar summary, not used by ZQE)
        log_std = torch.log(L_Sigma.diag()).expand_as(mu)
        return z, mu, log_std

    def loss(self, y, gllvm=None, **kwargs):
        """No parameters to optimise — return a zero loss."""
        dummy = next(self.gllvm.parameters())
        return torch.zeros(1, device=dummy.device, requires_grad=True), 0.0


class EncoderGaussianApprox(nn.Module):
    def __init__(self, gllvm, normalize=False):
        super().__init__()
        self.gllvm = gllvm
        self.normalize = normalize

        # Buffer for (X^T X)^{-1}, updated once per epoch by the fitter
        q = gllvm.wz.shape[1]
        self.register_buffer("inv_xtx", torch.eye(q))

    def forward(self, y, cols=None):
        """
        Linear MAP-like approximation to z | y.
        If cols is given, y contains only those selected features.
        """

        with torch.no_grad():

            # -----------------------------------------------------
            # Select appropriate loadings + bias block
            # -----------------------------------------------------
            if cols is None:
                X = self.gllvm.wz  # [p, q]
                bias = self.gllvm.bias  # [p]
            else:
                X = self.gllvm.wz[cols]  # [B, q]
                bias = None if self.gllvm.bias is None else self.gllvm.bias[cols]

            # -----------------------------------------------------
            # Approximate sufficient statistic log(1 + y) − bias
            # -----------------------------------------------------
            Y_approx = torch.log1p(y)
            if bias is not None:
                Y_approx = Y_approx - bias

            # -----------------------------------------------------
            # Solve linear system using precomputed inverse
            # XtY = Xᵀ Y    →    z = (Xᵀ X)⁻¹ Xᵀ Y
            # -----------------------------------------------------
            XtY = X.T @ Y_approx.T  # [q, n]
            Z_est = (self.inv_xtx @ XtY).T  # [n, q]

            # -----------------------------------------------------
            # Optional normalization: mean 0, std 1 per latent dim
            # -----------------------------------------------------
            if self.normalize:
                # TODO: normalize using a moving average!!!!! not per batch!!! (DOH!)
                mean = Z_est.mean(dim=0, keepdim=True)
                std = Z_est.std(dim=0, keepdim=True) + 1e-6
                Z_est = (Z_est - mean) / std
            else:
                Z_est *= 2.0

        # ---------------------------------------------------------
        # Empirical scaling factor (your design)
        # ---------------------------------------------------------
        return Z_est

    def sample(self, y, cols=None):
        z = self.forward(y, cols=cols)
        return z, z, None


def _fast_log_approx(X, Y, O=None):
    # X is [p, q]
    # Y is [n, p]
    # O is [n, 1] or None -> broadcasted

    Y_approx = torch.log1p(Y)
    if O is not None:
        Y_approx = Y_approx - O  # offset broadcasted

    # Solve linear system X B = Y_approx

    XtX = X.T @ X  # [q,q]  # same across all n
    XtY = X.T @ Y_approx.T  # [q,n]
    Z_est = torch.linalg.solve(XtX, XtY).T  # [n,q]
    return 2 * Z_est


# QUANTILE VERSION
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


class EncoderGaussianProjected(nn.Module):
    def __init__(self, gllvm, ema_momentum=0.05):
        super().__init__()
        self.gllvm = gllvm
        self.m = ema_momentum

        q = gllvm.wz.shape[1]
        self.register_buffer("inv_xtx", torch.eye(q))

        # running stats per latent dim (q dims)
        self.register_buffer("running_mean", torch.zeros(q))
        self.register_buffer("running_std", torch.ones(q))

        self.std_normal = Normal(0.0, 1.0)

    def _update_ema(self, z_batch):
        """
        z_batch: [n, q]
        update running mean/std per dimension
        """
        batch_mean = z_batch.mean(0)
        batch_std = z_batch.std(0) + 1e-6

        self.running_mean = (1 - self.m) * self.running_mean + self.m * batch_mean
        self.running_std = (1 - self.m) * self.running_std + self.m * batch_std

    def _project_gaussian(self, z_batch):
        """
        Project each latent dimension to N(0,1)
        z_batch: [n, q]
        """
        z_norm = (z_batch - self.running_mean) / self.running_std.clamp(min=1e-6)
        u = 0.5 * (1.0 + torch.erf(z_norm / torch.sqrt(torch.tensor(2.0))))
        u = u.clamp(1e-6, 1 - 1e-6)
        return self.std_normal.icdf(u)

    def forward(self, y, cols=None):
        """
        y: [n, p]
        output: [n, q]
        """

        # ---------------------------------------
        # Select X, bias: shapes depend on cols
        # ---------------------------------------
        if cols is None:
            X = self.gllvm.wz  # [p, q]
            bias = self.gllvm.bias  # [p]
        else:
            X = self.gllvm.wz[cols]  # [B, q]
            bias = None if self.gllvm.bias is None else self.gllvm.bias[cols]

        # ---------------------------------------
        # Compute Y_approx: [n, p]
        # ---------------------------------------
        Y_approx = torch.log1p(y)
        if bias is not None:
            Y_approx = Y_approx - bias  # broadcasted

        # ---------------------------------------
        # XtY = Xᵀ Yᵀ → [q, n]
        # ---------------------------------------
        XtY = X.T @ Y_approx.T  # [q, n]

        # z = (XᵀX)⁻¹ XtY  → [n, q]
        z = (self.inv_xtx @ XtY).T  # transpose to [n, q]

        # your original scaling
        z = 2.0 * z

        # update running distribution
        self._update_ema(z)

        # Gaussian projection
        z_proj = self._project_gaussian(z)

        return z_proj  # shape [n, q]

    def sample(self, y, cols=None):
        z = self.forward(y, cols)
        return z, z, None
