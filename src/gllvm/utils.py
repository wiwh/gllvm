# gllvm/utils/procrustes.py

import numpy as np
import torch
from scipy.linalg import orthogonal_procrustes
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


def to_np(x):
    """Convert torch tensor or list to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def procrustes_align_wz(W_true, W_est):
    """
    Align estimated Wz to true Wz via orthogonal Procrustes.

    GLLVM convention:
        Wz has shape (q, p)
            q = latent_dim
            p = number of observed responses

    Procrustes convention:
        Input matrices must be (p, q):
            rows = observed dims
            cols = latent dims

    We therefore transpose Wz before alignment:
        A = W_true.T  (p × q)
        B = W_est.T   (p × q)

    Solve:
        B R = A
        where R is an orthogonal (q × q) matrix

    Parameters
    ----------
    W_true : torch.Tensor, shape (q, p)
    W_est  : torch.Tensor, shape (q, p)

    Returns
    -------
    W_true_np : np.ndarray, shape (q, p)
        True loadings (unchanged)
    W_est_aligned : np.ndarray, shape (q, p)
        Estimated loadings after Procrustes alignment
    R : np.ndarray, shape (q, q)
        Orthogonal rotation/reflection matrix
    """

    # convert to numpy and transpose for Procrustes
    A = to_np(W_true)  # (p × q)
    B = to_np(W_est)  # (p × q)

    # handle q = 1 separately (sign ambiguity only)
    if A.shape[1] == 1:
        # choose R = +1 or -1 to match direction
        sign = np.sign((B * A).sum())
        R = np.array([[sign]], dtype=float)
        B_aligned = B @ R

        # return in original (q × p) format
        return A.T, B_aligned.T, R

    # general case q > 1
    R, _ = orthogonal_procrustes(B, A)
    B_aligned = B @ R

    # return in original (q × p) format
    return A, B_aligned, R
