import torch
import torch.optim as optim
import matplotlib.pyplot as plt


# ==========================================================
# Parameter + Latent History Recorder
# ==========================================================


class ParamHistory:
    def __init__(self, k_params=100, k_y=100):
        # how many parameter dims to track
        self.k_params = k_params

        # how many fixed y-samples to track
        self.k_y = k_y

        # tracked quantities
        self.wz = []
        self.bias = []
        self.wx = []
        self.z = []  # latent trajectories for fixed y-samples
        self.deviance = []

        # random index selections
        self.idx_wz = None
        self.idx_bias = None
        self.idx_wx = None

        self.idx_y = None  # fixed y-samples
        self.idx_z = None  # which latent dims to track

    # ------------------------------------------------------
    def _init_parameter_indices(self, model):
        """Select random parameter coordinates (once)."""
        if self.idx_wz is None:
            p = model.wz.numel()
            self.idx_wz = torch.randperm(p)[: min(self.k_params, p)]

        if self.idx_bias is None:
            p = model.bias.numel()
            self.idx_bias = torch.randperm(p)[: min(self.k_params, p)]

        if hasattr(model, "wx") and model.wx is not None and self.idx_wx is None:
            p = model.wx.numel()
            self.idx_wx = torch.randperm(p)[: min(self.k_params, p)]

    # ------------------------------------------------------
    def _init_y_indices(self, n_samples):
        """Choose fixed subset of y samples (once)."""
        if self.idx_y is None:
            self.idx_y = torch.randperm(n_samples)[: min(self.k_y, n_samples)]

    # ------------------------------------------------------
    def _init_z_indices(self, latent_dim):
        """Choose which latent dimensions to track (once)."""
        if self.idx_z is None:
            self.idx_z = torch.randperm(latent_dim)[: min(self.k_params, latent_dim)]

    # ------------------------------------------------------
    def record_params(self, model):
        """Record selected parameter coordinates."""
        self._init_parameter_indices(model)

        wz_flat = model.wz.detach().cpu().reshape(-1)
        self.wz.append(wz_flat[self.idx_wz].clone())

        bias_flat = model.bias.detach().cpu().reshape(-1)
        self.bias.append(bias_flat[self.idx_bias].clone())

        if self.idx_wx is not None:
            wx_flat = model.wx.detach().cpu().reshape(-1)
            self.wx.append(wx_flat[self.idx_wx].clone())

    # ------------------------------------------------------
    def record_latents(self, encoder, y, device):
        """
        For fixed y-samples, compute z = encoder(y0)
        and record selected latent coordinates.
        """
        n = len(y)
        self._init_y_indices(n)

        # fixed subset of y
        y0 = y[self.idx_y].to(device)

        with torch.no_grad():
            _, z0, _ = encoder.sample(y0)  # shape: (k_y, latent_dim)

        latent_dim = z0.shape[1]
        self._init_z_indices(latent_dim)

        # extract tracked latent dims and flatten
        # shape → (k_y * k_z,)
        z_sel = z0[:, self.idx_z].reshape(-1).cpu()
        self.z.append(z_sel.clone())

    def record_deviance(self, deviance):
        """Record deviance value."""
        self.deviance.append(deviance.unsqueeze(0).cpu())

    # ------------------------------------------------------
    def _plot_series(self, data, title):
        if len(data) == 0:
            print(f"[skip] no data to plot for {title}")
            return

        H = torch.stack(data)  # (epochs, tracked_dims)

        for j in range(H.shape[1]):
            plt.plot(H[:, j], linewidth=1)

        plt.title(title)
        plt.show()

    # ------------------------------------------------------
    def plot(self, model_true=None):
        self._plot_series(self.deviance, "Deviance")
        self._plot_series(self.wz, "Wz evolution")
        self._plot_series(self.bias, "Bias evolution")

        if len(self.wx) > 0:
            self._plot_series(self.wx, "Wx evolution")

        self._plot_series(self.z, "Latent evolution (fixed y-samples)")


# ==========================================================
# ZQE FITTER
# ==========================================================


class ZQEFitter:
    def __init__(
        self,
        gllvm,
        encoder,
        device="cpu",
        fit_encoder=True,
        lr_enc=1e-3,
        lr_dec=1e-3,
        args_enc=None,
        args_dec=None,
    ):
        self.gllvm = gllvm.to(device)
        self.encoder = encoder.to(device)
        self.device = device
        self.fit_encoder = fit_encoder
        self.args_enc = args_enc if args_enc is not None else {}
        self.args_dec = args_dec if args_dec is not None else {}

        if self.args_enc.get("lr", None) is None:
            self.args_enc["lr"] = lr_enc
        if self.args_dec.get("lr", None) is None:
            self.args_dec["lr"] = lr_dec

        scale_params = [gllvm.log_scale]
        no_scale_params = [p for n, p in gllvm.named_parameters() if n != "log_scale"]

        self.opt_enc = optim.Adam(
            list(self.encoder.parameters()) + scale_params, **self.args_enc
        )
        self.opt_dec = optim.Adam(no_scale_params, **self.args_dec)

        self.history = ParamHistory()

    # ------------------------------------------------------
    def encoder_step(self, y):

        loss, elbo = self.encoder.loss(y, self.gllvm)
        self.opt_enc.zero_grad()
        loss.backward()
        self.opt_enc.step()
        return elbo

    # ------------------------------------------------------
    def decoder_step(self, y):

        # Generate samples to compute expectation
        with torch.no_grad():
            zq0 = self.gllvm.sample_z(len(y))
            yq = self.gllvm.sample(z=zq0)

        # Generate conditional z
        with torch.no_grad():
            z, mu, _ = self.encoder.sample(y)
            z_q, mu_q, _ = self.encoder.sample(yq)

        # SHOULD WE TAKE THE MEAN OR Z??!! TODO

        m1 = self.gllvm.zq_log(y, z=z).sum(dim=-1).mean()
        m2 = self.gllvm.zq_log(yq, z=z_q).sum(dim=-1).mean()

        loss = -(m1 - m2)

        self.opt_dec.zero_grad()
        loss.backward()
        self.opt_dec.step()

    # ------------------------------------------------------
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

            # --- record both parameters and latent evolution ---
            self.history.record_params(self.gllvm)
            self.history.record_latents(self.encoder, y, self.device)

            print(f"Epoch {epoch+1}: ELBO={total_elbo:.2f}")

    # ------------------------------------------------------
    def plot_params(self):
        self.history.plot()


class VAEFitter:
    def __init__(self, gllvm, encoder, lr=1e-3, device="cpu"):
        self.gllvm = gllvm.to(device)
        self.encoder = encoder.to(device)
        self.device = device

        self.opt = optim.Adam(
            list(self.encoder.parameters()) + list(self.gllvm.parameters()),
            lr=lr,
        )

        self.history = ParamHistory()

    # ------------------------------------------------------
    def step(self, y):
        self.opt.zero_grad()
        loss, elbo = self.encoder.loss(y, self.gllvm)
        loss.backward()
        self.opt.step()
        return elbo

    # ------------------------------------------------------
    def fit(self, y, epochs=100, batch_size=512):
        y = y.to(self.device)
        n = len(y)

        for epoch in range(epochs):
            perm = torch.randperm(n, device=self.device)
            total_elbo = 0.0

            for i in range(0, n, batch_size):
                batch = y[perm[i : i + batch_size]]
                total_elbo += self.step(batch)

            # --- identical tracking as ZQE ---
            self.history.record_params(self.gllvm)
            self.history.record_latents(self.encoder, y, self.device)

            print(f"Epoch {epoch+1}: ELBO={total_elbo:.2f}")

    # ------------------------------------------------------
    def plot_params(self):
        self.history.plot()


# ==========================================================
# ZQE Gaussian Approx FITTER (BLOCKWISE VERSION)
# ==========================================================


import torch
import torch.optim as optim


class ZQEGAFitter:
    """
    Blockwise ZQ Estimator + Gaussian Approx Encoder.
    Completely scalable in p using feature blocks.
    """

    def __init__(
        self,
        gllvm,
        encoder,
        device="cpu",
        fit_encoder=True,
        lr_enc=1e-3,
        lr_dec=1e-3,
        args_enc=None,
        args_dec=None,
        feature_block=None,  # max number of features sampled per iteration
        scheduler=None,
    ):
        self.gllvm = gllvm.to(device)
        self.encoder = encoder.to(device)
        self.device = device
        self.fit_encoder = fit_encoder
        self.scheduler = scheduler

        if feature_block is None:
            feature_block = gllvm.wz.shape[0]  # all features
        self.feature_block = feature_block

        self.args_enc = args_enc if args_enc is not None else {}
        self.args_dec = args_dec if args_dec is not None else {}

        if self.args_enc.get("lr") is None:
            self.args_enc["lr"] = lr_enc
        if self.args_dec.get("lr") is None:
            self.args_dec["lr"] = lr_dec

        # Split scale param for encoder updates (your design)
        scale_params = [gllvm.log_scale]
        no_scale_params = [p for n, p in gllvm.named_parameters() if n != "log_scale"]

        self.opt_enc = optim.Adam(
            list(self.encoder.parameters()) + scale_params, **self.args_enc
        )
        self.opt_dec = optim.Adam(no_scale_params, **self.args_dec)

        self.history = ParamHistory()

    # ------------------------------------------------------
    # ENCODER UPDATE (blockwise)
    # ------------------------------------------------------
    def encoder_step(self, y_block):
        loss, elbo = self.encoder.loss(y_block, self.gllvm)
        self.opt_enc.zero_grad()
        loss.backward()
        self.opt_enc.step()
        return elbo

    # ------------------------------------------------------
    # DECODER UPDATE (blockwise ZQ)
    # ------------------------------------------------------
    def decoder_step(self, y_block, cols):
        mb = y_block.size(0)

        # Sample model-generated block
        with torch.no_grad():
            z0 = self.gllvm.sample_z(mb)
            yq_block = self.gllvm.sample_features(z0, cols)

        # Encode real + model blocks
        with torch.no_grad():
            _, z_real, _ = self.encoder.sample(y_block, cols=cols)
            _, z_q, _ = self.encoder.sample(yq_block, cols=cols)

        # Compute ZQ correction blockwise
        m1 = self.gllvm.zq_log_block(y_block, z_real, cols).sum(dim=-1).mean()
        m2 = self.gllvm.zq_log_block(yq_block, z_q, cols).sum(dim=-1).mean()

        loss = -(m1 - m2)

        self.opt_dec.zero_grad()
        loss.backward()
        self.opt_dec.step()

    # ------------------------------------------------------
    # FIT LOOP (blockwise)
    # ------------------------------------------------------

    def fit(self, y, epochs=100, batch_size=512):
        y = y.to(self.device)
        n, p = y.shape

        # TODO: make feature block an argument of fit
        B = min(p, self.feature_block)

        for epoch in range(epochs):
            # ==========================================================
            # 1. FULL XtX UPDATE (global geometry)
            # ==========================================================
            with torch.no_grad():
                X = self.gllvm.wz.detach()  # [p, q]
                XtX = X.T @ X  # [q, q]

                eps = 1e-4 * torch.mean(torch.diag(XtX))
                XtX_reg = XtX + eps * torch.eye(self.gllvm.q, device=X.device)
                inv = torch.linalg.inv(XtX_reg)

                self.encoder.inv_xtx.copy_(inv)

            # ==========================================================
            # 2. PER-EPOCH COLUMN PERMUTATION  (variance reduction)
            # ==========================================================
            perm_cols = torch.randperm(p, device=self.device)

            # ==========================================================
            # 3. PERMUTE ROWS
            # ==========================================================
            perm_rows = torch.randperm(n, device=self.device)
            total_elbo = 0.0

            # ==========================================================
            # 4. MINIBATCH ROW LOOP
            # ==========================================================
            for bi, start_row in enumerate(range(0, n, batch_size)):

                # -----------------------------
                # Row batch
                # -----------------------------
                row_idx = perm_rows[start_row : start_row + batch_size]
                y_rows = y[row_idx]  # [mb, p]
                mb = y_rows.size(0)

                # -----------------------------
                # Column block for this iteration
                # -----------------------------
                # Deterministic block index
                start_col = (bi * B) % p
                end_col = start_col + B

                if end_col <= p:
                    cols = perm_cols[start_col:end_col]
                else:
                    # Wrap around (rare; only at end of epoch)
                    cols = torch.cat([perm_cols[start_col:p], perm_cols[: end_col - p]])

                # -----------------------------
                # Extract feature block
                # -----------------------------
                y_block = y_rows[:, cols]  # [mb, B]

                # -----------------------------
                # ENCODER UPDATE
                # -----------------------------
                if self.fit_encoder:
                    total_elbo += self.encoder_step(y_block, cols=cols)

                # -----------------------------
                # DECODER UPDATE (ZQ correction)
                # -----------------------------
                self.decoder_step(y_block, cols)

            # ==========================================================
            # 5. RECORD TRAINING HISTORY
            # ==========================================================
            self.history.record_params(self.gllvm)
            self.history.record_latents(self.encoder, y, self.device)

            with torch.no_grad():
                z = self.encoder.forward(y)
                deviance = self.gllvm.deviance(y, z=z).sum()
                self.history.record_deviance(deviance)

            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step(deviance)

            print(f"Epoch {epoch+1}: Deviance={deviance:.2f}")

    def plot_params(self):
        self.history.plot()


class DevianceScheduler:
    def __init__(
        self,
        optimizer,
        factor=0.8,  # gentler reduction
        patience=30,  # require long plateau
        min_lr=1e-6,
        tol=5e-4,  # relative improvement threshold
        ema_alpha=0.1,  # smoothing over time
    ):
        """
        optimizer : decoder optimizer
        factor    : multiply lr by this on plateau
        patience  : epochs to wait after plateau
        tol       : minimal relative improvement
        ema_alpha : smoothing factor for EMA
        """
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.tol = tol
        self.ema_alpha = ema_alpha

        self.ema = None
        self.best = float("inf")
        self.wait = 0

    def step(self, dev):
        dev = float(dev)

        # ---------------------
        # 1. Update EMA
        # ---------------------
        if self.ema is None:
            self.ema = dev
        else:
            self.ema = self.ema_alpha * dev + (1 - self.ema_alpha) * self.ema

        # ---------------------
        # 2. Compute improvement wrt best EMA
        # ---------------------
        improvement = (self.best - self.ema) / (abs(self.best) + 1e-9)

        if improvement > self.tol:  # real progress
            self.best = self.ema
            self.wait = 0
        else:  # plateau-ish region
            self.wait += 1
            if self.wait >= self.patience:
                self._reduce_lr()
                self.wait = 0

    def _reduce_lr(self):
        for pg in self.optimizer.param_groups:
            old_lr = pg["lr"]
            new_lr = max(self.min_lr, old_lr * self.factor)
            pg["lr"] = new_lr
        print(f"[Scheduler] Reduced lr to {new_lr:.3e}")
