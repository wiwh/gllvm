import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.swa_utils import AveragedModel
from gllvm.utils import ParamHistory
from gllvm.glm_fit import initial_gaussian_fit, poisson_newton_batch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =======================================================
# Autoencoder + Correction
# =======================================================


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

        enc_params = list(self.encoder.parameters())
        if enc_params:
            self.opt_enc = optim.Adam(enc_params + scale_params, **self.args_enc)
            self.opt_dec = optim.Adam(no_scale_params, **self.args_dec)
        else:
            # Encoder has no learnable parameters (e.g. analytical MAP encoder);
            # put log_scale with the decoder optimizer to avoid empty/duplicate groups.
            self.opt_enc = None
            self.opt_dec = optim.Adam(no_scale_params + scale_params, **self.args_dec)

        self.history = ParamHistory()

    # ------------------------------------------------------
    def encoder_step(self, y):

        loss, elbo = self.encoder.loss(y, self.gllvm)
        if self.opt_enc is None or not loss.requires_grad:
            return elbo
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

        # Sample z ~ q(z|y).  We detach the encoder here (torch.no_grad), and
        # this is not an approximation: by the score-function identity the
        # gradient of the Z-estimating equation w.r.t. θ (decoder params) is
        #   ∂/∂θ E_q[T(y)·η(z;θ)] = E_q[T(y)·∂η/∂θ] + E_q[T(y)·η · ∂log q/∂θ]
        # The second term (score) is zero in expectation for any fixed y, so the
        # encoder chain-rule contribution cancels exactly.  Detaching is therefore
        # both valid and cheaper.
        with torch.no_grad():
            z, mu, _ = self.encoder.sample(y)
            z_q, mu_q, _ = self.encoder.sample(yq)

        m1 = self.gllvm.zq_log(y, z=z).sum(dim=-1).mean()
        m2 = self.gllvm.zq_log(yq, z=z_q).sum(dim=-1).mean()

        loss = -(m1 - m2)

        self.opt_dec.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.gllvm.parameters(), max_norm=5.0)
        if not torch.isnan(loss):
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


class ZQEZZZFitter:
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
        with torch.no_grad():
            zq0 = self.gllvm.sample_z(len(y))
            yq = self.gllvm.sample(z=zq0)

        z, mu, _ = self.encoder.sample(y)
        z_q, mu_q, _ = self.encoder.sample(yq)

        loss, elbo = self.encoder.loss(y, self.gllvm)

        loss_z = self.gllvm.zq_log(yq, z=z_q).sum(dim=-1).mean()
        loss_z += self.gllvm.zq_log(y, z=z).sum(dim=-1).mean()

        loss += loss_z

        self.opt_enc.zero_grad()
        loss.backward()
        self.opt_enc.step()
        return elbo

    # ------------------------------------------------------
    def decoder_step(self, y):

        # Generate conditional z
        with torch.no_grad():
            z, mu, _ = self.encoder.sample(y)

        # SHOULD WE TAKE THE MEAN OR Z??!! TODO

        # m1 = self.gllvm.zq_log(y, z=z).sum(dim=-1).mean()
        # m2 = self.gllvm.zq_log(yq, z=z_q).sum(dim=-1).mean()

        # loss = -(m1 - m2)

        loss = self.gllvm.log_prob(y, z=z).sum(dim=-1).mean()

        self.opt_dec.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.gllvm.parameters(), max_norm=5.0)
        if not torch.isnan(loss):
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
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.gllvm.parameters()),
            max_norm=5.0,
        )
        if not torch.isnan(loss):
            self.opt.step()
        return elbo

    # ------------------------------------------------------
    def fit(self, y, epochs=100, batch_size=512, patience=None, min_delta=1e-3,
            verbose=True):
        """
        Train VAE.

        Parameters
        ----------
        patience : int or None
            Early-stopping patience (epochs without improvement).
            None = no early stopping.
        min_delta : float
            Minimum ELBO improvement to count as progress.
        verbose : bool
            Print ELBO every epoch.
        """
        y = y.to(self.device)
        n = len(y)

        best_elbo = float("-inf")
        epochs_no_improve = 0

        for epoch in range(epochs):
            perm = torch.randperm(n, device=self.device)
            total_elbo = 0.0

            for i in range(0, n, batch_size):
                batch = y[perm[i : i + batch_size]]
                total_elbo += self.step(batch)

            # --- tracking ---
            self.history.record_params(self.gllvm)
            self.history.record_latents(self.encoder, y, self.device)

            if verbose:
                print(f"Epoch {epoch+1}: ELBO={total_elbo:.2f}")

            # --- early stopping ---
            if patience is not None:
                if total_elbo > best_elbo + min_delta:
                    best_elbo = total_elbo
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch+1} "
                                  f"(no improvement for {patience} epochs)")
                        break

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
    def encoder_step(self, y_block, cols=None):
        # cols is computed by the blockwise fit loop but the encoder
        # operates on the full feature vector, so it is unused here.
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
        new_lr = self.min_lr
        for pg in self.optimizer.param_groups:
            old_lr = pg["lr"]
            new_lr = max(self.min_lr, old_lr * self.factor)
            pg["lr"] = new_lr
        print(f"[Scheduler] Reduced lr to {new_lr:.3e}")


# ==========================================================
# ZQE Poisson Fitter  (stacked-Newton E-step + Adam M-step)
# ==========================================================

class ZQEPoissonFitter:
    """
    ZQE training loop for Poisson (or log1p-Poisson) GLLVMs.

    E-step: batched Newton MAP for observed + n_mc fantasy samples,
            solved jointly via ``poisson_newton_batch`` (fully vectorised).
    M-step: one Adam step on the ZQE objective with the Z values fixed.
    LR:     ``torch.optim.lr_scheduler.ReduceLROnPlateau`` on mean per-epoch
            gradient norm — reduces LR when gradient norms plateau.

    Parameters
    ----------
    gllvm : GLLVM
        The decoder model (parameters to be optimised).
    device : str
        'cuda' or 'cpu'.
    lr : float
        Initial Adam learning rate.
    min_lr : float
        Floor for ``ReduceLROnPlateau``.
    patience : int
        ``ReduceLROnPlateau`` patience (epochs).
    factor : float
        LR reduction factor (default 0.5).
    threshold : float
        ``ReduceLROnPlateau`` threshold for "no improvement".
    n_mc : int
        Number of fantasy samples per epoch.
    ema_decay : float
        EMA decay for the model-averaged snapshot.
    ema_start : int
        Epoch at which EMA averaging begins.
    """

    def __init__(
        self,
        gllvm,
        device: str = "cpu",
        lr: float = 1e-0,
        min_lr: float = 1e-8,
        patience: int = 50,
        factor: float = 0.5,
        threshold: float = 1e-3,
        n_mc: int = 4,
        ema_decay: float = 0.95,
        ema_start: int = 20,
    ):
        self.gllvm     = gllvm.to(device)
        self.device    = device
        self.n_mc      = n_mc
        self.ema_start = ema_start

        self.opt = optim.SGD(list(self.gllvm.parameters()), lr=lr)
        logging.info(f"Initialized ZQE SGD Poisson Fitter with lr={lr}, min_lr={min_lr}, "
                     f"patience={patience}, factor={factor}, threshold={threshold}, "
                     f"n_mc={n_mc}, ema_decay={ema_decay}, ema_start={ema_start}")

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode="min", factor=factor,
            patience=patience, threshold=threshold, min_lr=min_lr,
        )

        ema_fn = lambda avg, cur, n: ema_decay * avg + (1 - ema_decay) * cur
        self.ema_model  = AveragedModel(self.gllvm, avg_fn=ema_fn)
        self.ema_active = False

        # history lists filled by fit()
        self.history = ParamHistory()
        self.h_loss  = []
        self.h_gnorm = []
        self.h_lr    = []
        self.h_eval  = []   # filled if eval_fn is passed to fit()

    # ----------------------------------------------------------
    def _e_step(self, y: torch.Tensor, idx=None, n_sim: int | None = None):
        """
        Newton MAP for observed data + fantasy samples.

        Parameters
        ----------
        y     : (N_total, p) full count tensor
        idx   : 1-D index tensor selecting the observed mini-batch rows.
                ``None`` means use all rows.
        n_sim : total number of fantasy samples.  ``None`` defaults to
                ``len(idx)`` (same size as the observed batch).

        Returns
        -------
        z_obs   : (B, q)        MAP latents for the observed batch
        yq_list : list of 1 tensor of shape (n_sim, p)  — fantasy data
        zq_list : list of 1 tensor of shape (n_sim, q)  — fantasy MAP latents
        """
        y_batch = y if idx is None else y[idx]                 # (B, p)
        B       = len(y_batch)
        if n_sim is None:
            n_sim = B

        W = self.gllvm.wz.detach()                             # (p, q)
        b = (self.gllvm.bias.detach()
             if self.gllvm.bias is not None
             else torch.zeros(self.gllvm.p, device=self.device))  # (p,)

        yq = self.gllvm.sample(z=self.gllvm.sample_z(n_sim))  # (n_sim, p)

        Y_obs_t = y_batch.float().T.contiguous()               # (p, B)
        Yq_t    = yq.float().T.contiguous()                    # (p, n_sim)
        Y_stack = torch.cat([Y_obs_t, Yq_t], dim=1)           # (p, B+n_sim)
        offset  = b.unsqueeze(1).expand_as(Y_stack).contiguous()

        B0       = initial_gaussian_fit(W, Y_stack, offset=offset)
        B_hat, _ = poisson_newton_batch(
            X=W, Y=Y_stack, B0=B0, offset=offset,
            lam=1.0, max_iter=30, max_halvings=10, tol=1e-6, verbose=False,
        )
        Z_hat   = B_hat.T.contiguous()                         # (B+n_sim, q)
        z_obs   = Z_hat[:B]
        zq_list = [Z_hat[B:]]
        yq_list = [yq]
        return z_obs, yq_list, zq_list

    # ----------------------------------------------------------
    def _m_step(self, y_batch: torch.Tensor, z_obs, yq_list, zq_list):
        """One Adam step on the ZQE objective with fixed Z."""
        m1 = self.gllvm.zq_log(y_batch, z=z_obs).sum(-1).mean()
        m2 = torch.stack([
            self.gllvm.zq_log(yq_list[k], z=zq_list[k]).sum(-1).mean()
            for k in range(len(yq_list))
        ]).mean()
        loss = -(m1 - m2)

        self.opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(list(self.gllvm.parameters()), 5.0).item()
        if not torch.isnan(loss):
            self.opt.step()
        return loss.item(), gn

    # ----------------------------------------------------------
    def fit(
        self,
        y: torch.Tensor,
        epochs: int = 1500,
        batch_size: int | None = None,
        sim_factor: int | None = None,
        verbose: bool = True,
    ):
        """
        Run the ZQE training loop.

        Parameters
        ----------
        y          : (N, p) count tensor
        epochs     : number of outer iterations
        batch_size : observed mini-batch size.  ``None`` → full batch (N).
        sim_factor : fantasy size = ``sim_factor * batch_size``.
                     ``None`` → fantasy size equals ``batch_size`` (1×).
        verbose    : print progress every 300 epochs

        Returns
        -------
        self  (for chaining)
        """
        y  = y.to(self.device)
        N  = len(y)
        bs = N if batch_size is None else int(batch_size)
        # n_sim per mini-batch: None passes through to _e_step (defaults to bs)
        n_sim = None if sim_factor is None else int(sim_factor * bs)

        for ep in range(epochs):
            perm = torch.randperm(N, device=self.device)

            batch_losses, batch_gns = [], []
            for start in range(0, N, bs):
                idx = perm[start: start + bs]

                with torch.no_grad():
                    z_obs, yq_list, zq_list = self._e_step(y, idx=idx, n_sim=n_sim)

                y_batch = y[idx]
                loss_val, gn = self._m_step(y_batch, z_obs, yq_list, zq_list)
                batch_losses.append(loss_val)
                batch_gns.append(gn)

            loss_val = float(np.mean(batch_losses))
            gn       = float(np.mean(batch_gns))

            prev_lr = self.opt.param_groups[0]["lr"]
            self.scheduler.step(gn)
            new_lr  = self.opt.param_groups[0]["lr"]

            self.h_loss.append(loss_val)
            self.h_gnorm.append(gn)
            self.h_lr.append(new_lr)
            self.history.record_params(self.gllvm)

            if ep >= self.ema_start:
                self.ema_model.update_parameters(self.gllvm)
                self.ema_active = True

            if verbose:
                if new_lr < prev_lr:
                    print(f"    ↓ lr {prev_lr:.2e}→{new_lr:.2e}"
                          f"  gnorm={gn:.4f}  ep={ep+1}")
                if (ep + 1) % 100 == 0:
                    print(f"  ep {ep+1:4d}/{epochs}"
                          f"  loss={loss_val:+.4f}"
                          f"  gnorm={gn:.4f}"
                          f"  lr={new_lr:.2e}")

        return self

    # ----------------------------------------------------------
    @property
    def model(self):
        """Return EMA model if active, else the raw decoder."""
        return self.ema_model.module if self.ema_active else self.gllvm


# ==========================================================
# ZQE Poisson Fitter  (stacked-Newton E-step + L-BFGS M-step)
# ==========================================================

class ZQELBFGSFitter:
    """
    EM-style ZQE training for Poisson GLLVMs using L-BFGS for the M-step.

    Algorithm (per outer iteration)
    --------------------------------
    E-step : draw ``n_sim`` fantasy samples, run batched Newton MAP for
             all ``N + n_sim`` columns simultaneously (same solver as
             ``ZQEPoissonFitter``).  Z tensors are detached — fixed constants
             for the M-step.
    M-step : run ``torch.optim.LBFGS`` with a deterministic closure on the
             fixed Z tensors.  The closure is safe to call multiple times
             (Wolfe line search) because Z does not change within the M-step.

    Advantages over Adam
    ---------------------
    * No learning-rate to tune — L-BFGS uses its own Wolfe line search.
    * Each M-step drives W to near-convergence on the current E-step targets,
      so far fewer outer iterations are needed (~100–200 vs ~1500).
    * Gradient clipping not required (L-BFGS has built-in damping).

    Parameters
    ----------
    gllvm        : GLLVM decoder (parameters to optimise).
    device       : 'cuda' or 'cpu'.
    n_sim        : number of fantasy samples per outer step.
    lbfgs_iters  : ``max_iter`` for ``torch.optim.LBFGS``.
    refresh_every: re-draw fantasy samples every this many outer steps.
                   1 (default) = refresh every step (maximum exploration).
    ema_decay    : EMA decay for the model snapshot.
    ema_start    : outer iteration at which EMA begins.
    """

    def __init__(
        self,
        gllvm,
        device: str = "cpu",
        n_sim: int = 10,
        lbfgs_iters: int = 20,
        refresh_every: int = 1,
        lambda_w: float = 1e-2,
        ema_decay: float = 0.95,
        ema_start: int = 10,
    ):
        self.gllvm         = gllvm.to(device)
        self.device        = device
        self.n_sim         = n_sim
        self.lbfgs_iters   = lbfgs_iters
        self.refresh_every = refresh_every
        self.lambda_w      = lambda_w
        self.ema_start     = ema_start

        self.opt = optim.LBFGS(
            list(self.gllvm.parameters()),
            lr=0.1,
            max_iter=lbfgs_iters,
            line_search_fn="strong_wolfe",
        )

        ema_fn = lambda avg, cur, n: ema_decay * avg + (1 - ema_decay) * cur
        self.ema_model  = AveragedModel(self.gllvm, avg_fn=ema_fn)
        self.ema_active = False

        self.h_loss  = []
        self.h_gnorm = []

    # ----------------------------------------------------------
    @torch.no_grad()
    def _e_step(self, y: torch.Tensor, n_sim: int):
        """Newton MAP for N observed + n_sim fantasy columns.

        Decorated with ``@torch.no_grad()`` — imputation and Z solving are
        never tracked.  All returned tensors are therefore detached constants
        for the M-step, preserving the ZQE cancellation property exactly.
        """
        N = len(y)
        W = self.gllvm.wz.detach()
        b = (self.gllvm.bias.detach()
             if self.gllvm.bias is not None
             else torch.zeros(self.gllvm.p, device=self.device))

        yq = self.gllvm.sample(z=self.gllvm.sample_z(n_sim))

        Y_obs_t = y.float().T.contiguous()
        Yq_t    = yq.float().T.contiguous()
        Y_stack = torch.cat([Y_obs_t, Yq_t], dim=1)
        offset  = b.unsqueeze(1).expand_as(Y_stack).contiguous()

        B0       = initial_gaussian_fit(W, Y_stack, offset=offset)
        B_hat, _ = poisson_newton_batch(
            X=W, Y=Y_stack, B0=B0, offset=offset,
            lam=1.0, max_iter=30, max_halvings=10, tol=1e-6, verbose=False,
        )
        Z_hat = B_hat.T.contiguous()
        return Z_hat[:N], yq, Z_hat[N:]

    # ----------------------------------------------------------
    def _make_closure(self, y_obs, z_obs, yq, zq):
        """Return a closure that L-BFGS can call multiple times.

        Uses the full log-likelihood log p(y|z;W) so the objective is concave
        in W (for exponential families).  A small L2 penalty ``lambda_w`` on W
        ensures the problem is strictly convex and L-BFGS converges to a
        bounded solution even when obs and fantasy Hessians nearly cancel.
        """
        def closure():
            self.opt.zero_grad()
            with torch.enable_grad():
                try:
                    m1   = self.gllvm.log_prob(y_obs, z=z_obs).sum(-1).mean()
                    m2   = self.gllvm.log_prob(yq,    z=zq   ).sum(-1).mean()
                    reg  = 0.5 * self.lambda_w * (self.gllvm.wz ** 2).sum()
                    loss = -(m1 - m2) + reg
                    if not torch.isfinite(loss):
                        raise ValueError("non-finite loss")
                    loss.backward()
                except Exception:
                    # NaN/Inf from line-search overshoot — return large finite
                    # value so L-BFGS backtracks.
                    loss = torch.tensor(1e6, device=self.device, dtype=torch.float32)
            return loss
        return closure

    # ----------------------------------------------------------
    def fit(self, y: torch.Tensor, n_outer: int = 200, verbose: bool = True):
        """
        Run the EM L-BFGS training loop.

        Parameters
        ----------
        y       : (N, p) count tensor
        n_outer : number of outer EM iterations
        verbose : print progress every 20 iterations

        Returns self.
        """
        y = y.to(self.device)
        z_obs = yq = zq = None   # cached E-step results

        for outer in range(n_outer):
            # ── E-step (refresh on schedule) ──────────────────────────────
            if z_obs is None or (outer % self.refresh_every == 0):
                z_obs, yq, zq = self._e_step(y, self.n_sim)

            # ── M-step ────────────────────────────────────────────────────
            closure = self._make_closure(y, z_obs, yq, zq)
            loss_tensor = self.opt.step(closure)
            loss_val = loss_tensor.item() if loss_tensor is not None else float("nan")

            gn = max(
                (p.grad.norm().item() for p in self.gllvm.parameters()
                 if p.grad is not None),
                default=0.0,
            )

            self.h_loss.append(loss_val)
            self.h_gnorm.append(gn)

            if outer >= self.ema_start:
                self.ema_model.update_parameters(self.gllvm)
                self.ema_active = True

            if verbose and (outer + 1) % 20 == 0:
                print(f"  outer {outer+1:4d}/{n_outer}"
                      f"  loss={loss_val:+.4f}"
                      f"  gnorm={gn:.4f}")

        return self

    # ----------------------------------------------------------
    @property
    def model(self):
        """Return EMA model if active, else the raw decoder."""
        return self.ema_model.module if self.ema_active else self.gllvm
