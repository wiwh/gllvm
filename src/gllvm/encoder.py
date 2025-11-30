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

        return nll, -nll.item()


class EncoderMAP(nn.Module):
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
        z = self.forward(y)

        # prior log p(z)
        lpz = -0.5 * (z**2).sum(dim=-1)

        # decoder likelihood log p(y|z)
        linpar = gllvm.forward(z)
        logp_ygz = gllvm.log_prob(y, linpar=linpar).sum(dim=-1)

        # MAP = maximize (log p(y|z) + log p(z))
        loss = -(logp_ygz + lpz).mean()
        return loss, -loss.item()
