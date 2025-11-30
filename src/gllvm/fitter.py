import torch
import torch.optim as optim
import matplotlib.pyplot as plt


# ==========================================================
# Parameter History Recorder
# ==========================================================


class ParamHistory:
    def __init__(self):
        self.wz = []
        self.bias = []
        self.wx = []

    def record(self, model):
        self.wz.append(model.wz.detach().cpu().clone())
        self.bias.append(model.bias.detach().cpu().clone())
        if hasattr(model, "wx") and model.wx is not None:
            self.wx.append(model.wx.detach().cpu().clone())

    def _plot_param(self, hist, true=None, title=""):
        if len(hist) == 0:
            print(f"[skip] no values to plot for {title}")
            return

        H = torch.stack(hist)  # (epochs, p, latent_dim?)

        for j in range(H.shape[1]):
            plt.plot(H[:, j].reshape(-1), label=f"{title}[{j}]")
            if true is not None:
                plt.axhline(true[j].item(), color="k", linestyle="--")

        plt.title(f"{title} evolution")
        plt.legend()
        plt.show()

    def plot(self, model_true=None):
        if model_true is not None:
            true_wz = model_true.wz.detach().cpu()
            true_bias = model_true.bias.detach().cpu()
        else:
            true_wz = true_bias = None

        self._plot_param(self.wz, true_wz, "Wz")
        self._plot_param(self.bias, true_bias, "Bias")

        if len(self.wx) > 0:
            true_wx = (
                model_true.wx.detach().cpu()
                if (model_true and model_true.wx is not None)
                else None
            )
            self._plot_param(self.wx, true_wx, "Wx")


# ==========================================================
# ZQE FITTER
# ==========================================================


class ZQEFitter:
    def __init__(self, gllvm, encoder, lr=1e-3, device="cpu", fit_encoder=True):
        self.gllvm = gllvm.to(device)
        self.encoder = encoder.to(device)
        self.device = device
        self.fit_encoder = fit_encoder

        scale_params = [gllvm.log_scale]
        no_scale_params = [p for n, p in gllvm.named_parameters() if n != "log_scale"]

        self.opt_enc = optim.Adam(list(self.encoder.parameters()) + scale_params, lr=lr)
        self.opt_dec = optim.Adam(no_scale_params, lr=lr)

        # parameter tracking
        self.history = ParamHistory()

    # ----------------------------------------------------------
    def encoder_step(self, y):
        self.opt_enc.zero_grad()
        loss, elbo = self.encoder.loss(y, self.gllvm)
        loss.backward()
        self.opt_enc.step()
        return elbo

    # ----------------------------------------------------------
    def decoder_step(self, y):
        self.opt_dec.zero_grad()

        with torch.no_grad():
            z, _, _ = self.encoder.sample(y)

        with torch.no_grad():
            zq0 = self.gllvm.sample_z(len(y))
            yq = self.gllvm.sample(z=zq0)
            zq, _, _ = self.encoder.sample(yq)

        m1 = self.gllvm.zq_log(y, z=z).sum(dim=-1).mean()
        m2 = self.gllvm.zq_log(yq, z=zq).sum(dim=-1).mean()
        loss = -(m1 - m2)

        loss.backward()
        self.opt_dec.step()

    # ----------------------------------------------------------
    def fit(self, y, epochs=100, batch_size=512):
        y = y.to(self.device)
        n = len(y)

        for epoch in range(epochs):
            perm = torch.randperm(n, device=self.device)
            total_elbo = 0.0

            for i in range(0, n, batch_size):
                batch = y[perm[i : i + batch_size]]
                if self.fit_encoder:
                    total_elbo += self.encoder_step(batch)
                self.decoder_step(batch)

            # record params per epoch
            self.history.record(self.gllvm)

            print(f"Epoch {epoch+1}: ELBO={total_elbo:.2f}")

    # ----------------------------------------------------------
    def plot_params(self, true_model=None):
        self.history.plot(true_model)


# ==========================================================
# VAE FITTER
# ==========================================================


class VAEFitter:
    def __init__(self, gllvm, encoder, lr=1e-3, device="cpu"):
        self.gllvm = gllvm.to(device)
        self.encoder = encoder.to(device)
        self.device = device

        # ONE optimizer for ALL parameters
        self.opt = optim.Adam(
            list(self.encoder.parameters()) + list(self.gllvm.parameters()), lr=lr
        )

        self.history = ParamHistory()

    # ----------------------------------------------------------
    def step(self, y):
        self.opt.zero_grad()

        loss, elbo = self.encoder.loss(y, self.gllvm)
        loss.backward()

        self.opt.step()
        return elbo

    # ----------------------------------------------------------
    def fit(self, y, epochs=100, batch_size=512):
        y = y.to(self.device)
        n = len(y)

        for epoch in range(epochs):
            perm = torch.randperm(n, device=self.device)
            total_elbo = 0.0

            for i in range(0, n, batch_size):
                batch = y[perm[i : i + batch_size]]
                total_elbo += self.step(batch)

            self.history.record(self.gllvm)
            print(f"Epoch {epoch+1}: ELBO={total_elbo:.2f}")

    # ----------------------------------------------------------
    def plot_params(self, true_model=None):
        self.history.plot(true_model)
