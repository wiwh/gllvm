import torch
from torch import nn


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
