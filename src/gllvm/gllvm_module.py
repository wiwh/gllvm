"""
GLLVM module with support for heterogeneous GLM families.

This module defines:
    - GLMFactory: a wrapper that stores a GLM family, response indices,
      and optional per-family parameters.
    - GLLVM: a latent variable model over p response variables, each of
      which uses a user-specified GLM family.

Design:
    * All GLM classes accept (linpar, scale).
    * The GLLVM stores a universal scale parameter of length p.
      GLM families that do not use scale simply ignore it.
    * Sampling works by slicing linpar and scale by GLM-specific indices
      and instantiating the corresponding GLM distributions.

This design ensures consistency with exponential-family structure and
provides a clean interface for ZQ-based estimators.
"""

import torch
import torch.nn as nn
from typing import Any, List

from gllvm.glms import GLMMixin


# ============================================================
# GLM Factory
# ============================================================


import torch
import numpy as np
from typing import Any, List, Iterable


class GLMFamily:
    """
    Stores a GLM class, the response indices it applies to, and
    optional keyword parameters passed at construction time.

    Parameters
    ----------
    GLM : type[GLMMixin]
        A GLM class (e.g., PoissonGLM, GammaGLM, etc.)
    idx : list[int], range, numpy array, or torch.Tensor
        Indices of response variables modeled by this GLM.
    params : dict or None
        Optional fixed keyword arguments supplied to GLM constructor.
    name : str or None
        Optional name for display.
    """

    def __init__(
        self,
        GLM: type,
        idx: Any,
        params: dict[str, Any] | None,
        name: str | None,
    ):
        self.GLM = GLM
        self.params = params if params is not None else {}
        self.name = name

        # ---- Robust idx conversion ----
        self.idx = self._to_index_tensor(idx)

    def _to_index_tensor(self, idx: Any) -> torch.Tensor:
        """
        Convert idx (list, range, ndarray, tensor) into a torch.long tensor.
        """
        if isinstance(idx, torch.Tensor):
            return idx.long()

        if isinstance(idx, range):
            return torch.tensor(list(idx), dtype=torch.long)

        if isinstance(idx, np.ndarray):
            return torch.tensor(idx.astype(np.int64), dtype=torch.long)

        if isinstance(idx, (list, tuple)):
            return torch.tensor(idx, dtype=torch.long)

        raise TypeError(f"Unsupported idx type: {type(idx)}")

    def __call__(self, **kwargs):
        merged = {**self.params, **kwargs}
        return self.GLM(**merged)

    def __repr__(self):
        idx_list = self.idx.tolist()
        n = len(idx_list)

        if n <= 10:
            idx_str = str(idx_list)
        else:
            first = ", ".join(map(str, idx_list[:5]))
            last = ", ".join(map(str, idx_list[-5:]))
            idx_str = f"[{first}, ..., {last}] (total {n})"

        return (
            f"GLMFamily(GLM={self.GLM.__name__}, "
            f"idx={idx_str}, params={self.params}, name={self.name})"
        )


# ============================================================
# GLLVM Model
# ============================================================


class GLLVM(nn.Module):
    """
    Generalized Latent Variable Model.

    Each response variable i = 1,...,p is associated with a GLM family.
    Latent variables z shape: (n_samples, q)
    Covariates x shape: (n_samples, k)

    Model:
        linpar = z @ wz + x @ wx + bias
        y_i | z, x ~ GLM_i( linpar_i, scale_i )

    Parameters
    ----------
    latent_dim : int
        Dimension q of latent variables.
    output_dim : int
        Number p of response variables.
    feature_dim : int
        Number k of covariates.
    bias : bool
        Whether to include a learned bias vector of size p.
    """

    def __init__(
        self, latent_dim: int, output_dim: int, feature_dim: int = 0, bias: bool = True
    ):
        super().__init__()

        self.q = latent_dim
        self.p = output_dim
        self.k = feature_dim

        # Loadings for latent variables
        self.wz = nn.Parameter(torch.randn(self.q, self.p))

        # Loadings for covariates
        self.wx = (
            nn.Parameter(torch.randn(self.k, self.p) * 0.5) if self.k > 0 else None
        )

        # Optional per-response intercept
        self.bias = nn.Parameter(torch.zeros(self.p)) if bias else None

        # Universal scale parameter (dispersion)
        self.log_scale = nn.Parameter(torch.zeros(self.p))

        # GLM assignment list
        self.families: List[GLMFamily] = []
        self.p_defined = 0

    # --------------------------------------------------------
    # Building the model
    # --------------------------------------------------------
    @property
    def scale(self) -> torch.Tensor:
        return torch.exp(torch.clamp(self.log_scale, -10, 10))

    def add_glm(
        self,
        GLM: type[GLMMixin],
        idx: List[int] | torch.Tensor,
        params: dict[str, Any] | None = None,
        name: str | None = None,
    ):
        """
        Assign a GLM family to a set of response indices.
        """
        family = GLMFamily(GLM, idx, params, name)
        self.families.append(family)
        self.p_defined += len(family.idx)

    def to(self, *args, **kwargs):
        """
        Overrides nn.Module.to() to also move GLM family indices.
        """
        new_self = super().to(*args, **kwargs)
        device = next(new_self.parameters()).device
        for fam in new_self.families:
            fam.idx = fam.idx.to(device)
        return new_self

    def _check_assignments(self):
        """
        Ensure that every response variable has exactly one GLM family.
        """
        if self.p_defined != self.p:
            raise ValueError(
                f"Total assigned responses is {self.p_defined}, but expected {self.p}."
            )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    def forward(self, z: torch.Tensor, x: torch.Tensor | None = None):
        """
        Compute linear predictor.

        Shapes:
            z: (n_samples, q)
            x: (n_samples, k) or None
        """
        self._check_assignments()

        linpar = z @ self.wz

        if self.k > 0:
            if x is None:
                raise ValueError("Covariates x must be provided when num_covar > 0.")
            linpar = linpar + x @ self.wx

        if self.bias is not None:
            linpar = linpar + self.bias

        return linpar

    # --------------------------------------------------------
    # Sampling
    # --------------------------------------------------------

    def sample_z(self, num_samples: int):
        device = self.wz.device
        return torch.randn((num_samples, self.q), device=device)

    def _check_sample_input(
        self,
        num_samples: int,
        x: torch.Tensor | None,
        z: torch.Tensor | None,
    ):
        if z is not None and z.shape[1] != self.q:
            raise ValueError(f"z has {z.shape[1]} latent dims but expected {self.q}.")

        if self.k > 0:
            if x is None:
                raise ValueError("x must be provided when num_covar > 0.")
            if x.shape != (num_samples, self.k):
                raise ValueError(
                    f"x shape is {x.shape}, expected ({num_samples}, {self.k})."
                )

    # Proxy method calls to GLM blocks

    def sample(
        self,
        num_samples: int | None = None,
        z: torch.Tensor | None = None,
        x: torch.Tensor | None = None,
    ):
        """
        Sample y ~ p(y | z, x) using the assigned GLM families.
        """
        self._check_assignments()

        if num_samples is None and z is not None:
            num_samples = z.shape[0]
        elif num_samples is None and x is not None:
            num_samples = x.shape[0]
        elif num_samples is None:
            raise ValueError("Either num_samples, z, or x must be provided.")

        if z is None:
            z = self.sample_z(num_samples)

        self._check_sample_input(num_samples, x, z)

        linpar = self.forward(z, x)

        y = torch.empty((num_samples, self.p), device=linpar.device)

        # Sample block-wise by GLM family
        for glm in self.families:
            idx = glm.idx
            lp_slice = linpar[:, idx]
            sc_slice = self.scale[idx]
            dist = glm(linpar=lp_slice, scale=sc_slice)
            y[:, idx] = dist.sample()

        return y

    def mean(self, *, linpar=None, z=None, x=None):
        """
        Compute the mean of each response under its GLM family.

        Exactly one of the following must be provided:
            - linpar (tensor of shape [n_samples, p])
            - z (tensor), optionally x

        Parameters
        ----------
        linpar : torch.Tensor | None
            Precomputed linear predictor.
        z : torch.Tensor | None
            Latent variables (n_samples, q).
        x : torch.Tensor | None
            Covariates (n_samples, k), optional.

        Returns
        -------
        mean : torch.Tensor
            Tensor of shape (n_samples, p)
        """
        self._check_assignments()

        # ---------------------------
        # Determine linpar
        # ---------------------------
        if linpar is not None:
            # Directly provided
            pass

        elif z is not None:
            # Compute from model
            linpar = self.forward(z, x)

        else:
            raise ValueError(
                "mean() requires one of: linpar=..., or z=... (x optional)."
            )

        # ---------------------------
        # Evaluate mean blockwise
        # ---------------------------
        result = torch.empty_like(linpar)

        for glm in self.families:
            idx = glm.idx
            lp_slice = linpar[:, idx]
            sc_slice = self.scale[idx]

            dist = glm(linpar=lp_slice, scale=sc_slice)
            result[:, idx] = dist.mean()

        return result

    def log_prob(self, y, *, linpar=None, z=None, x=None):
        """
        Compute log p(y | z, x) or log p(y | linpar).

        Exactly one of the following must be provided:
            - linpar
            - z (and optionally x)
        """

        self._check_assignments()

        # ---- Determine linpar ----
        if linpar is not None:
            pass  # user gave linear predictor
        elif z is not None:
            linpar = self.forward(z, x)
        else:
            raise ValueError(
                "log_prob() requires either linpar=... or z=... (with optional x)."
            )

        # ---- Compute log-prob ----
        logp = torch.zeros(y.shape[0], self.p, device=y.device)

        for fam in self.families:
            idx = fam.idx
            lp = linpar[:, idx]
            sc = self.scale[idx]

            dist = fam(linpar=lp, scale=sc)
            logp[:, idx] = dist.log_prob(y[:, idx])

        return logp

    def zq_log(self, y, *, linpar=None, z=None, x=None):
        """
        Compute log p(y | z, x) or log p(y | linpar).

        Exactly one of the following must be provided:
            - linpar
            - z (and optionally x)
        """

        self._check_assignments()

        # ---- Determine linpar ----
        if linpar is not None:
            pass  # user gave linear predictor
        elif z is not None:
            linpar = self.forward(z, x)
        else:
            raise ValueError(
                "log_prob() requires either linpar=... or z=... (with optional x)."
            )

        # ---- Compute log-prob ----
        logp = torch.zeros(y.shape[0], self.p, device=y.device)

        for fam in self.families:
            idx = fam.idx
            lp = linpar[:, idx]
            sc = self.scale[idx]

            dist = fam(linpar=lp, scale=sc)
            logp[:, idx] = dist.zq_log(y[:, idx])

        return logp

    # --------------------------------------------------------
    # Representation
    # --------------------------------------------------------

    def __repr__(self):
        glm_str = ",\n  ".join(str(g) for g in self.families)
        return (
            f"GLLVM(\n"
            f"  q={self.q}, p={self.p}, k={self.k},\n"
            f"  families=[\n  {glm_str}\n  ]\n"
            f")"
        )


if __name__ == "__main__":
    from gllvm.glms import PoissonGLM, GammaGLM, NegativeBinomialGLM

    torch.manual_seed(0)

    print("Building GLLVM...")
    model = GLLVM(latent_dim=2, output_dim=6000, feature_dim=1, bias=True)

    # Assign GLM families
    model.add_glm(PoissonGLM, idx=range(0, 2000), name="Poisson block")
    model.add_glm(GammaGLM, idx=range(2000, 4000), name="Gamma block")
    model.add_glm(NegativeBinomialGLM, idx=range(4000, 6000), name="NB block")

    print("Model structure:")
    print(model)
    print()

    n = 5
    # Create synthetic latent variables z
    z = model.sample_z(n)

    # Create synthetic covariates x
    x = torch.randn(n, 1)

    # Sample latent variables
    print("Sampled z:")
    print(z)
    print()

    # Forward to compute linear predictor
    linpar = model.forward(z, x)
    print("Linear predictor:")
    print(linpar)
    print()

    # Sample from the model
    with torch.no_grad():
        y = model.sample(num_samples=n, z=z, x=x)
    print("Sampled y:")
    print(y)
    print()

    print("Test completed successfully.")
