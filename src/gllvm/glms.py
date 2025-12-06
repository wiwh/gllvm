"""
ZQ-GLM wrappers over PyTorch distributions.

Each GLM defines:
    - eta() : natural parameter (GLM linear predictor)
    - T(y)  : sufficient statistic
    - deviance(y) : classical GLM deviance
    - __init__: converts (linpar, scale) into parameters required by
                torch.distributions

Each GLM inherits from the corresponding torch.distributions class but exposes
a clean GLM interface suitable for GLLVMs and ZQ estimators.
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

    @abstractmethod
    def deviance(self, y: torch.Tensor) -> torch.Tensor:
        """
        Classical GLM deviance contribution.
        Must return a tensor of same shape as y.
        """
        pass

    def zq_log(self, y: torch.Tensor) -> torch.Tensor:
        """
        Informative part of complete-data log-likelihood.
        """
        return self.T(y) * self.eta()


# ============================================================
# Gaussian GLM
# ============================================================


class GaussianGLM(D.Normal, GLMMixin):
    """
    Gaussian GLM with identity link.
    Var[Y] = scale (dispersion phi)
    """

    def __init__(self, linpar: torch.Tensor, scale: float | torch.Tensor = 1.0):
        self.linpar = linpar
        self.scale = torch.as_tensor(scale, device=linpar.device)
        std = torch.sqrt(self.scale)
        super().__init__(loc=linpar, scale=std)

    def eta(self):
        return self.linpar

    def T(self, y):
        return y

    def deviance(self, y):
        # Gaussian deviance (up to constant factor) = (y - mu)^2 / variance
        mu = self.linpar
        var = self.scale
        return (y - mu) ** 2 / var

    def __repr__(self):
        return f"GaussianGLM(linpar={self.linpar}, scale={self.scale})"


# ============================================================
# Poisson GLM
# ============================================================


class PoissonGLM(D.Poisson, GLMMixin):
    """
    Poisson GLM with log link.
    Var[Y] = mu, scale fixed to 1.
    """

    def __init__(self, linpar: torch.Tensor, scale: float | torch.Tensor = 1.0):
        self.linpar = linpar
        self.scale = torch.tensor(1.0, device=linpar.device)
        rate = torch.exp(linpar)
        super().__init__(rate=rate)

    def eta(self):
        return self.linpar

    def T(self, y):
        return y

    def deviance(self, y, eps=1e-8):
        """
        D = 2 * [ y*log(y/mu) - (y - mu) ].
        """
        mu = self.rate  # = exp(linpar)
        y_safe = y.clamp_min(eps)
        mu_safe = mu.clamp_min(eps)
        return 2 * (y_safe * torch.log(y_safe / mu_safe) - (y_safe - mu_safe))

    def __repr__(self):
        return f"PoissonGLM(linpar={self.linpar})"


# ============================================================
# Gamma GLM
# ============================================================


class GammaGLM(D.Gamma, GLMMixin):
    """
    Gamma GLM with log link.
    Var[Y] = scale * mu^2
    scale = dispersion parameter
    """

    def __init__(self, linpar: torch.Tensor, scale: float | torch.Tensor = 1.0):
        self.linpar = linpar
        self.scale = torch.as_tensor(scale, device=linpar.device)

        mu = torch.exp(linpar)
        concentration = 1.0 / self.scale  # shape = alpha
        rate = concentration / mu  # beta = alpha / mu

        super().__init__(concentration=concentration, rate=rate)

    def eta(self):
        return self.linpar

    def T(self, y):
        return torch.log(y)

    def deviance(self, y, eps=1e-8):
        """
        D = 2 * [ (y - mu)/mu - log(y/mu) ]
        """
        mu = torch.exp(self.linpar)
        y_safe = y.clamp_min(eps)
        mu_safe = mu.clamp_min(eps)

        return 2 * ((y_safe - mu_safe) / mu_safe - torch.log(y_safe / mu_safe))

    def __repr__(self):
        return f"GammaGLM(linpar={self.linpar}, scale={self.scale})"


# ============================================================
# Negative Binomial GLM (NB2)
# ============================================================


class NegativeBinomialGLM(D.NegativeBinomial, GLMMixin):
    """
    Negative Binomial GLM with log link.
    Var[Y] = mu + scale * mu^2
    total_count = 1/scale
    """

    def __init__(self, linpar: torch.Tensor, scale: float | torch.Tensor = 1.0):
        self.linpar = linpar
        self.scale = torch.as_tensor(scale, device=linpar.device)

        mu = torch.exp(linpar)
        theta = self.scale.clamp_min(1e-8)
        total_count = 1.0 / theta
        probs = total_count / (total_count + mu)

        super().__init__(total_count=total_count, probs=probs)

    def eta(self):
        return self.linpar

    def T(self, y):
        return y

    def deviance(self, y, eps=1e-8):
        """
        NB2 deviance:
        D = 2 * [ y log(y/mu) - (y+θ) log((y+θ)/(mu+θ)) ]
        """
        mu = torch.exp(self.linpar)
        theta = self.scale.clamp_min(eps)

        y_safe = y.clamp_min(eps)
        mu_safe = mu.clamp_min(eps)

        return 2 * (
            y_safe * torch.log(y_safe / mu_safe)
            - (y_safe + theta) * torch.log((y_safe + theta) / (mu_safe + theta))
        )

    def __repr__(self):
        return f"NegativeBinomialGLM(linpar={self.linpar}, scale={self.scale})"


# ============================================================
# Binomial GLM
# ============================================================


class BinomialGLM(D.Binomial, GLMMixin):
    """
    Binomial GLM with total_count n and logit link.
    Var[Y] = n p (1-p)
    """

    def __init__(
        self,
        linpar: torch.Tensor,
        total_count: Any = 1,
        scale: float | torch.Tensor = 1.0,
    ):
        self.linpar = linpar
        self.total_count = torch.as_tensor(total_count, device=linpar.device)
        self.scale = torch.tensor(1.0, device=linpar.device)  # unused
        super().__init__(total_count=self.total_count, logits=linpar)

    def eta(self):
        return self.linpar

    def T(self, y):
        return y

    def deviance(self, y, eps=1e-8):
        """
        D = 2 [ y log(y/mu) + (n-y) log((n-y)/(n-mu)) ]
        """
        n = self.total_count
        p = self.probs
        mu = n * p

        y_safe = y.clamp_min(eps)
        mu_safe = mu.clamp_min(eps)
        ny_safe = (n - y).clamp_min(eps)
        nmu_safe = (n - mu).clamp_min(eps)

        return 2 * (
            y_safe * torch.log(y_safe / mu_safe)
            + ny_safe * torch.log(ny_safe / nmu_safe)
        )

    def __repr__(self):
        return f"BinomialGLM(linpar={self.linpar}, total_count={self.total_count})"
