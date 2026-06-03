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
        self.wz = nn.Parameter(torch.randn(self.p, self.q))

        # Loadings for covariates
        self.wx = (
            nn.Parameter(torch.randn(self.p, self.k) * 0.5) if self.k > 0 else None
        )

        # Optional per-response intercept
        self.bias = nn.Parameter(torch.zeros(self.p)) if bias else None

        # Universal scale parameter (dispersion)
        self.log_scale = nn.Parameter(torch.zeros(self.p))

        # GLM assignment list
        self.families: List[GLMFamily] = []
        self.p_defined = 0

        # Optional structural-zero mask on loadings (p, q).  1 = free, 0 = forced zero.
        # Registered as a buffer so it travels with .to(device) / state_dict.
        self.register_buffer("wz_mask", torch.ones(self.p, self.q))

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

    def set_wz_mask(self, mask: torch.Tensor):
        """
        Set a structural-zero mask on the loadings matrix.

        Parameters
        ----------
        mask : torch.Tensor, shape (p, q) or (q,) broadcastable to (p, q)
            Binary mask.  Entry (i, j) == 0 means response i does not load on
            latent dimension j.  The mask is stored as a buffer and is NOT a
            learnable parameter.

        Example
        -------
        # silence the last 3 latent dimensions for all responses:
        mask = torch.ones(model.p, model.q)
        mask[:, -3:] = 0
        model.set_wz_mask(mask)
        """
        mask = mask.float().to(self.wz.device)
        if mask.shape != (self.p, self.q):
            raise ValueError(
                f"mask shape {tuple(mask.shape)} does not match (p={self.p}, q={self.q})"
            )
        self.wz_mask = mask

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

        wz_eff = self.wz * self.wz_mask          # apply structural-zero mask
        linpar = z @ wz_eff.T

        if self.k > 0:
            if x is None:
                raise ValueError("Covariates x must be provided when num_covar > 0.")
            linpar = linpar + x @ self.wx.T

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

    def sample_features(self, z, cols, x=None):
        """
        Blockwise sampling for only a subset of features.

        z:    [n, q]
        cols: [B] feature indices
        x:    optional covariates
        """

        self._check_assignments()

        n = z.shape[0]
        device = z.device

        # Compute full linpar but extract block
        linpar_full = self.forward(z, x)  # [n, p]
        linpar = linpar_full[:, cols]  # [n, B]

        # Pre-select bias and scale block
        bias_block = None
        if self.bias is not None:
            bias_block = self.bias[cols]  # [B]

        scale_block = self.scale[cols]  # [B]

        # Allocate result
        y_block = torch.empty((n, len(cols)), device=device)

        # For each GLM family, sample only overlapping part
        for fam in self.families:
            idx = fam.idx  # [p_fam]

            # intersection between cols and fam.idx
            mask = torch.isin(cols, idx)
            if not mask.any():
                continue

            # Actual subset of columns belonging to this family
            cols_sub = cols[mask]
            lp_sub = linpar[:, mask]  # [n, num_sub]

            # Scale + bias per family subset
            sc_sub = scale_block[mask]

            dist = fam(linpar=lp_sub, scale=sc_sub)
            y_block[:, mask] = dist.sample()

        return y_block

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
            result[:, idx] = dist.mean

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

    def mean_block(self, z, cols, x=None):
        linpar_full = self.forward(z, x)
        linpar = linpar_full[:, cols]

        if self.bias is not None:
            linpar = linpar + self.bias[cols]

        scale_block = self.scale[cols]

        out = torch.empty_like(linpar)

        for fam in self.families:
            idx = fam.idx
            mask = torch.isin(cols, idx)
            if not mask.any():
                continue

            lp_sub = linpar[:, mask]
            sc_sub = scale_block[mask]

            dist = fam(linpar=lp_sub, scale=sc_sub)
            out[:, mask] = dist.mean()

        return out

    def zq_log_block(self, y_block, z, cols, x=None):
        self._check_assignments()

        linpar_full = self.forward(z, x)
        linpar = linpar_full[:, cols]

        scale_block = self.scale[cols]
        logp = torch.zeros_like(y_block)

        for fam in self.families:
            idx = fam.idx

            mask = torch.isin(cols, idx)
            if not mask.any():
                continue

            lp_sub = linpar[:, mask]
            y_sub = y_block[:, mask]
            sc_sub = scale_block[mask]

            dist = fam(linpar=lp_sub, scale=sc_sub)
            logp[:, mask] = dist.zq_log(y_sub)

        return logp

    def deviance(self, y, *, z=None, linpar=None, x=None):
        """
        Compute total deviance for all features.

        Parameters
        ----------
        y : (n, p)
        z : (n, q) optional
        linpar : (n, p) optional
        x : (n, k) optional

        Returns
        -------
        dev : scalar tensor
        """
        self._check_assignments()

        # Determine linpar
        if linpar is None:
            if z is None:
                raise ValueError("Provide either linpar or z.")
            linpar = self.forward(z, x)

        # Compute model means once (broadcasted)
        mu = self.mean(linpar=linpar)

        total = 0.0

        # Loop over GLM families
        for fam in self.families:
            idx = fam.idx
            lp_slice = linpar[:, idx]
            mu_slice = mu[:, idx]
            y_slice = y[:, idx]
            sc_slice = self.scale[idx]

            glm = fam(linpar=lp_slice, scale=sc_slice)
            total += glm.deviance(y_slice).sum()

        return total

    def deviance_block(self, y_block, z_block, cols, *, x_block=None):
        """
        Compute deviance on a COLUMN BLOCK only.

        Parameters
        ----------
        y_block : (mb, B)
            minibatch of responses with selected features
        z_block : (mb, q)
            minibatch latent variables
        cols : (B,)
            feature indices corresponding to the columns of y_block
        x_block : optional (mb, k)

        Returns
        -------
        dev : scalar tensor
        """
        self._check_assignments()

        # Compute linear predictor for selected columns
        lin_full = self.forward(z_block, x_block)  # (mb, p)
        lin_block = lin_full[:, cols]  # (mb, B)

        # Means for selected columns
        mu_full = self.mean(linpar=lin_full)  # (mb, p)
        mu_block = mu_full[:, cols]  # (mb, B)

        total = 0.0

        # Loop over GLM families
        for fam in self.families:
            fam_idx = fam.idx

            # mask: which of 'cols' belong to this GLM family?
            mask = torch.isin(cols, fam_idx)
            if not mask.any():
                continue  # skip non-overlapping families

            # Map global → block-local feature positions
            block_cols = mask.nonzero().squeeze(-1)

            y_slice = y_block[:, block_cols]
            lp_slice = lin_block[:, block_cols]
            mu_slice = mu_block[:, block_cols]
            sc_slice = self.scale[cols[block_cols]]

            glm = fam(linpar=lp_slice, scale=sc_slice)
            total += glm.deviance(y_slice).sum()

        return total

    def initialize(self, y, eps=1e-3):
        """
        Initialize per-feature bias so that the model mean matches the empirical mean.
        GLM-family aware: uses the correct intercept rule for each block.
        """

        y = y.to(self.wz.device)
        for fam in self.families:
            idx = fam.idx
            y_col = y[:, idx]

            # Empirical mean per feature
            m = y_col.mean(0).clamp_min(eps)

            if self.bias is not None:
                if fam.GLM.__name__ == "PoissonGLM":
                    self.bias.data[idx] = torch.log(m)

                elif fam.GLM.__name__ == "GammaGLM":
                    self.bias.data[idx] = torch.log(m)

                elif fam.GLM.__name__ == "NegativeBinomialGLM":
                    self.bias.data[idx] = torch.log(m)

                elif fam.GLM.__name__ == "BinomialGLM":
                    total = fam.params.get("total_count", 1)
                    p = (m / total).clamp(1e-5, 1 - 1e-5)
                    self.bias.data[idx] = torch.log(p / (1 - p))

                else:
                    # fallback: mean-matching through canonical link
                    self.bias.data[idx] = torch.log(m)

        print("✓ GLLVM bias initialized from data.")

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
