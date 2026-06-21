"""
Covariance kernels for the GP-GLLVM latent prior.

The GP-GLLVM places an *independent* Gaussian process on each of the ``q`` latent
factors (``B = I``), so the joint latent covariance is **block-diagonal** across
factors.  A :class:`Kernel` produces the ``q`` per-factor covariance blocks
``K_k(coords)`` and their Cholesky factors, exploiting that block structure: every
operation is ``K x K`` per factor, never ``(qK) x (qK)``.

Shapes
------
Coordinates ``coords`` have shape ``(*batch, K, d)`` (``d`` = input/coordinate
dimension: 1 for time, 2 for space, ...).  ``blocks`` returns ``(*batch, q, K, K)``.
"""

import torch
import torch.nn as nn


class Kernel(nn.Module):
    """Base class: per-factor covariance over input coordinates (``B = I``).

    Subclasses implement :meth:`blocks`; :meth:`cholesky` is derived.  A kernel maps
    coordinates ``(*batch, K, d)`` to the ``q`` per-factor blocks ``(*batch, q, K, K)``.
    """

    def blocks(self, coords: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def cholesky(self, coords: torch.Tensor) -> torch.Tensor:
        """Lower-Cholesky factor of each per-factor block, ``(*batch, q, K, K)``."""
        return torch.linalg.cholesky(self.blocks(coords))


class RBFKernel(Kernel):
    r"""Squared-exponential kernel with a **per-factor** length-scale.

    .. math::
        K_k(t, t') = \exp\!\big(-\|t - t'\|^2 / (2\,\ell_k^2)\big), \qquad k = 1,\dots,q

    Distinct length-scales across factors are what make the loadings identifiable in
    the GP-GLLVM (the kernels break the rotation gauge), so each factor carries its
    own ``ell_k``; they are stored on the log-scale and learned by the fitter.

    Parameters
    ----------
    latent_dim : int
        Number of latent factors ``q`` (one length-scale each).
    lengthscale : float or 1-D tensor of shape ``(q,)``
        Initial length-scale(s).  A scalar is broadcast to all ``q`` factors.
    jitter : float
        Added to the diagonal of each block for numerical positive-definiteness.
    """

    def __init__(self, latent_dim: int, lengthscale=1.0, jitter: float = 1e-4):
        super().__init__()
        self.q = latent_dim
        self.jitter = jitter
        ls = torch.as_tensor(lengthscale, dtype=torch.get_default_dtype())
        if ls.ndim == 0:
            ls = ls.expand(latent_dim).clone()
        if ls.shape != (latent_dim,):
            raise ValueError(
                f"lengthscale must be a scalar or shape ({latent_dim},), got {tuple(ls.shape)}"
            )
        self.log_lengthscale = nn.Parameter(ls.log())

    @property
    def lengthscale(self) -> torch.Tensor:
        return self.log_lengthscale.exp()

    def blocks(self, coords: torch.Tensor) -> torch.Tensor:
        # coords (*batch, K, d) -> squared distances (*batch, K, K)
        d2 = ((coords.unsqueeze(-2) - coords.unsqueeze(-3)) ** 2).sum(-1)
        ell = self.lengthscale.view(self.q, 1, 1)                       # (q, 1, 1)
        Kq = torch.exp(-0.5 * d2.unsqueeze(-3) / ell ** 2)             # (*batch, q, K, K)
        K = coords.shape[-2]
        eye = torch.eye(K, dtype=coords.dtype, device=coords.device)
        return Kq + self.jitter * eye

    def __repr__(self):
        ls = [round(x, 3) for x in self.lengthscale.detach().cpu().tolist()]
        return f"RBFKernel(q={self.q}, lengthscale={ls}, jitter={self.jitter})"
