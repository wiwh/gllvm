"""
ZQ-GLM wrappers over PyTorch distributions.

Each GLM defines:
    - eta() : natural parameter (GLM linear predictor)
    - T(y)  : sufficient statistic
    - __init__: converts (linpar, scale) into the parameters required by
                torch.distributions (Normal, Poisson, Gamma, etc.)

Each GLM inherits from the corresponding torch.distributions class to expose
standard parameterization and sampling methods.

Users can easily create new GLMs (e.g. with nonstandard link functions) by subclassing
the appropriate distribution class and implementing the required methods.
"""

from abc import ABC, abstractmethod
import torch
import torch.distributions as D
from typing import Any

__all__ = [
    "GLMMixin",
    "GaussianGLM",
    "PoissonGLM",
    "GammaGLM",
    "NegativeBinomialGLM",
    "BinomialGLM",
]


# ============================================================
# Abstract Mixin
# ============================================================


class GLMMixin(ABC):
    linpar: torch.Tensor
    scale: float | torch.Tensor

    @abstractmethod
    def eta(self) -> torch.Tensor:
        """Natural parameter eta."""
        pass

    @abstractmethod
    def T(self, y: torch.Tensor) -> torch.Tensor:
        """Sufficient statistic T(y)."""
        pass

    def zq_log(self, y: torch.Tensor) -> torch.Tensor:
        """Informative part of complete-data log-likelihood."""
        return self.T(y) * self.eta() / self.scale


# ============================================================
# Gaussian GLM
# ============================================================


class GaussianGLM(D.Normal, GLMMixin):
    """Gaussian GLM with identity link.
    Var[Y] = scale (dispersion phi)
    """

    def __init__(self, linpar: torch.Tensor, scale: float | torch.Tensor = 1.0):
        self.linpar = linpar
        self.scale = torch.as_tensor(scale, device=linpar.device)
        std = torch.sqrt(self.scale)
        super().__init__(loc=linpar, scale=std)

    def eta(self) -> torch.Tensor:
        return self.linpar

    def T(self, y: torch.Tensor) -> torch.Tensor:
        return y

    def __repr__(self):
        return f"GaussianGLM(linpar={self.linpar}, scale={self.scale})"


# ============================================================
# Poisson GLM
# ============================================================


class PoissonGLM(D.Poisson, GLMMixin):
    """Poisson GLM with log link.
    Var[Y] = mu, scale fixed internally to 1.
    Remarks: the scale argument is needed for standardization.
    """

    def __init__(self, linpar: torch.Tensor, scale: float | torch.Tensor = 1.0):
        self.linpar = linpar
        self.scale = torch.tensor(1.0, device=linpar.device)
        rate = torch.exp(linpar)
        super().__init__(rate=rate)

    def eta(self) -> torch.Tensor:
        return self.linpar

    def T(self, y: torch.Tensor) -> torch.Tensor:
        return y

    def __repr__(self):
        return f"PoissonGLM(linpar={self.linpar})"


# ============================================================
# Gamma GLM
# ============================================================


class GammaGLM(D.Gamma, GLMMixin):
    """Gamma GLM with log link.
    Var[Y] = scale * mu^2, shape = 1/scale
    """

    def __init__(self, linpar: torch.Tensor, scale: float | torch.Tensor = 1.0):
        self.linpar = linpar
        self.scale = torch.as_tensor(scale, device=linpar.device)

        mu = torch.exp(linpar)
        concentration = 1.0 / self.scale  # shape
        rate = concentration / mu  # beta = alpha / mu

        super().__init__(concentration=concentration, rate=rate)

    def eta(self) -> torch.Tensor:
        return self.linpar

    def T(self, y: torch.Tensor) -> torch.Tensor:
        return torch.log(y)

    def __repr__(self):
        return f"GammaGLM(linpar={self.linpar}, scale={self.scale})"


# ============================================================
# Negative Binomial GLM
# ============================================================


class NegativeBinomialGLM(D.NegativeBinomial, GLMMixin):
    """Negative Binomial GLM with log link.
    Var[Y] = mu + scale * mu^2, total_count = 1/scale
    """

    def __init__(self, linpar: torch.Tensor, scale: float | torch.Tensor = 1.0):
        self.linpar = linpar
        self.scale = torch.as_tensor(scale, device=linpar.device)

        mu = torch.exp(linpar)
        total_count = 1.0 / self.scale  # r
        probs = total_count / (total_count + mu)  # p

        super().__init__(total_count=total_count, probs=probs)

    def eta(self) -> torch.Tensor:
        return self.linpar

    def T(self, y: torch.Tensor) -> torch.Tensor:
        return y

    def __repr__(self):
        return f"NegativeBinomialGLM(linpar={self.linpar}, scale={self.scale})"


# ============================================================
# Binomial GLM
# ============================================================


class BinomialGLM(D.Binomial, GLMMixin):
    """Binomial GLM with logit link.
    Var[Y] = n * p * (1-p)
    Remarks: the scale argument is needed for standardization.
    """

    def __init__(
        self,
        linpar: torch.Tensor,
        total_count: Any = 1,
        scale: float | torch.Tensor = 1.0,
    ):
        self.linpar = linpar
        self.total_count = total_count
        self.scale = torch.tensor(1.0, device=linpar.device)  # placeholder only
        super().__init__(total_count=total_count, logits=linpar)

    def eta(self) -> torch.Tensor:
        return self.linpar

    def T(self, y: torch.Tensor) -> torch.Tensor:
        return y

    def __repr__(self):
        return f"BinomialGLM(linpar={self.linpar}, total_count={self.total_count})"
