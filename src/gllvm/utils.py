# gllvm/utils/procrustes.py

import numpy as np
import torch
from scipy.linalg import orthogonal_procrustes


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
    A = to_np(W_true).T  # (p × q)
    B = to_np(W_est).T  # (p × q)

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
    return A.T, B_aligned.T, R
