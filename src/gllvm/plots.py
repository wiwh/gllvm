# gllvm/plots.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import orthogonal_procrustes

from gllvm.utils import procrustes_align_wz  # aligns Wz directly


# ==========================================================
# utilities
# ==========================================================


def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def align_latent(Z_true, Z_est):
    """
    Procrustes rotation so Z_est aligns to Z_true.
    """
    Zt = to_np(Z_true)
    Ze = to_np(Z_est)

    Zt0 = Zt - Zt.mean(axis=0, keepdims=True)
    Ze0 = Ze - Ze.mean(axis=0, keepdims=True)

    R, _ = orthogonal_procrustes(Ze0, Zt0)
    Ze_aligned = Ze0 @ R

    return Zt0, Ze_aligned, R


def plot_scatter(true, est, title):
    t = to_np(true).ravel()
    e = to_np(est).ravel()

    mn = min(t.min(), e.min())
    mx = max(t.max(), e.max())

    plt.figure(figsize=(5, 5))
    plt.scatter(t, e, s=25)
    plt.plot([mn, mx], [mn, mx], "k--", lw=1)
    plt.xlabel("true")
    plt.ylabel("estimated")
    plt.title(title)
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ==========================================================
# main comparisons
# ==========================================================


def compare_wz(true_model, est_model):
    """
    Compare Wz with Procrustes alignment:
        W_est * R ≈ W_true
    """
    W_true = true_model.wz
    W_est = est_model.wz

    A, B_aligned, R = procrustes_align_wz(W_true, W_est)

    # Frobenius alignment error
    err = np.linalg.norm(to_np(A) - to_np(B_aligned))

    title = f"Wz (Procrustes aligned) — Frobenius error = {err:.4f}"
    plot_scatter(A, B_aligned, title=title)


def compare_bias(true_model, est_model):
    plot_scatter(true_model.bias, est_model.bias, title="Bias: true vs estimated")


def compare_wx(true_model, est_model):
    if not hasattr(true_model, "wx") or true_model.wx is None:
        print("[compare_wx] model has no wx")
        return
    plot_scatter(true_model.wx, est_model.wx, title="Wx: true vs estimated")


def compare_z(Z_true, Z_est):
    """
    Latent comparison using Procrustes alignment.
    """
    Zt, Ze, R = align_latent(Z_true, Z_est)

    # Compute alignment error
    err = np.linalg.norm(Zt - Ze)

    k = Zt.shape[1]

    if k == 1:
        title = f"Latent z (1D, aligned) — Frobenius error = {err:.4f}"
        plot_scatter(Zt, Ze, title)
        return

    # Multi-dimensional: show dim 0 vs dim 0
    plt.figure(figsize=(5, 5))
    plt.scatter(Zt[:, 0], Ze[:, 0], s=20)
    mn = min(Zt[:, 0].min(), Ze[:, 0].min())
    mx = max(Zt[:, 0].max(), Ze[:, 0].max())
    plt.plot([mn, mx], [mn, mx], "k--", lw=1)
    plt.xlabel("true z (dim 0)")
    plt.ylabel("estimated z (dim 0 aligned)")
    plt.title(f"Latent z: dim0 comparison (aligned) — error = {err:.4f}")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ==========================================================
# combined API
# ==========================================================


def plot_model_comparison(true_model, est_model, Z_true=None, Z_est=None):
    compare_wz(true_model, est_model)
    compare_bias(true_model, est_model)
    compare_wx(true_model, est_model)

    if Z_true is not None and Z_est is not None:
        compare_z(Z_true, Z_est)
