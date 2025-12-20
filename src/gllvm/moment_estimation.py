import torch
import torch.nn as nn
import torch.optim as optim
from gllvm.utils import ParamHistory

# ==========================================================
# SIMPLE DETERMINISTIC ENCODER
# ==========================================================


class SimpleDetEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.out = nn.Linear(hidden, latent_dim)

    def deterministic(self, y):
        return self.out(self.net(y))

    def sample(self, y):
        z = self.deterministic(y)
        return z, z, None


# ==========================================================
# LOSS FUNCTIONS
# ==========================================================


def recon_loss(encoder, y, gllvm):
    """
    θ-loss and φ-loss component.
    No detach. Prior penalty on z for stability.
    """
    z = encoder.deterministic(y)
    logpy = gllvm.log_prob(y, z=z).sum(dim=-1).mean()
    prior_penalty = 0.5 * (z**2).mean()
    return -(logpy) + prior_penalty


def encode_and_score_terms(encoder, gllvm, y_sim, device):
    """
    Compute T(Y_sim) and μ(Z_sim; θ_detached).
    """
    z_sim = encoder.deterministic(y_sim)

    # ---- η = Wz + b  (θ detached) ----
    with torch.no_grad():
        lin = z_sim @ gllvm.wz.detach().T
        if gllvm.bias is not None:
            lin = lin + gllvm.bias.detach()

        mu = gllvm.mean(linpar=lin)

    # ---- Sufficient statistics T(Y_sim) ----
    T_y = torch.zeros_like(y_sim)
    for fam in gllvm.families:
        idx = fam.idx
        lp = lin[:, idx]
        sc = gllvm.scale[idx]
        glm = fam(linpar=lp, scale=sc)
        T_y[:, idx] = glm.T(y_sim[:, idx])

    return T_y, mu


def moment_loss_phi(encoder, gllvm, batch_size, device):
    """
    φ-loss: enforce   E[T(Y)] = E[ μ(Z) ].
    θ is *detached* everywhere => no Jacobians.
    """
    with torch.no_grad():
        z0 = gllvm.sample_z(batch_size)
        y_sim = gllvm.sample(z=z0).to(device)

    T_y, mu = encode_and_score_terms(encoder, gllvm, y_sim, device)

    LHS = T_y.mean(dim=0)  # E[T(Y)]
    RHS = mu.mean(dim=0)  # E[μ(Z)]

    return ((LHS - RHS) ** 2).sum()


# ==========================================================
# Z-ESTIMATION FITTER (CORRECT)
# ==========================================================


class ZEstFitter:
    """
    Correct Z-estimation training loop (the only stable one):

        φ-step: minimize   recon_loss + λ * moment_loss
               (θ detached == no Jacs)

        θ-step: minimize   -log pθ(y | zφ(y) )
               (z detached => classical Z-est update)

    """

    def __init__(
        self,
        gllvm,
        encoder,
        device="cpu",
        lr_phi=1e-3,
        lr_theta=1e-3,
        moment_weight=1.0,
    ):

        self.gllvm = gllvm.to(device)
        self.encoder = encoder.to(device)
        self.device = device
        self.moment_weight = moment_weight

        # Independent optimizers
        self.opt_phi = optim.Adam(self.encoder.parameters(), lr=lr_phi)
        self.opt_theta = optim.Adam(self.gllvm.parameters(), lr=lr_theta)

        self.history = ParamHistory()

    # ------------------------------------------------------
    # φ-step (encoder update)
    # ------------------------------------------------------
    def encoder_step(self, batch):
        batch = batch.to(self.device)
        bs = batch.size(0)

        loss_rec = recon_loss(self.encoder, batch, self.gllvm)
        loss_mom = moment_loss_phi(self.encoder, self.gllvm, bs, self.device)

        loss = loss_rec + self.moment_weight * loss_mom

        self.opt_phi.zero_grad()
        loss.backward()
        self.opt_phi.step()

        return loss_rec.item(), loss_mom.item()

    # ------------------------------------------------------
    # θ-step (decoder update)
    # ------------------------------------------------------
    def decoder_step(self, batch):
        batch = batch.to(self.device)

        with torch.no_grad():
            z = self.encoder.deterministic(batch)

        z = z.detach()
        logpy = self.gllvm.log_prob(batch, z=z).sum(dim=-1).mean()
        loss = -logpy

        self.opt_theta.zero_grad()
        loss.backward()
        self.opt_theta.step()

        return loss.item()

    # ------------------------------------------------------
    # TRAINING LOOP
    # ------------------------------------------------------
    def fit(self, y, epochs=100, batch_size=512):
        y = y.to(self.device)
        n = len(y)

        for epoch in range(epochs):
            perm = torch.randperm(n, device=self.device)
            total_rec = total_mom = total_dec = 0.0

            for i in range(0, n, batch_size):
                batch = y[perm[i : i + batch_size]]

                # φ-step
                rec, mom = self.encoder_step(batch)
                total_rec += rec
                total_mom += mom

                # θ-step
                dec = self.decoder_step(batch)
                total_dec += dec

            # ---- Record evolution ----
            self.history.record_params(self.gllvm)
            self.history.record_latents(self.encoder, y, self.device)

            with torch.no_grad():
                z = self.encoder.deterministic(y)
                dev = self.gllvm.deviance(y, z=z).sum()
            self.history.record_deviance(dev)

            print(
                f"Epoch {epoch+1:03d}: "
                f"Recon={total_rec:.2f}  "
                f"Moment={total_mom:.2f}  "
                f"Decoder={total_dec:.2f}  "
                f"Deviance={float(dev):.2f}"
            )

    def plot(self):
        self.history.plot()
