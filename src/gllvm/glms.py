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
    "PoissonLog1pGLM",
    "PoissonMixedTGLM",
    "PoissonSqrtGLM",
    "PoissonLog1pSqrtGLM",
    "PoissonMultiTGLM",
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
        rate = torch.exp(torch.clamp(linpar, max=10.0))  # clamp: exp(10)≈22k, avoids inf
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


class PoissonLog1pGLM(PoissonGLM):
    """
    Poisson GLM with log link, but using T(y) = log(1+y) as the sufficient
    statistic in the ZQE estimating equation instead of the canonical T(y) = y.

    This is a valid (non-canonical) choice: the ZQE centring

        E[T(y) * eta(z_q(y))] - E_model[T(yq) * eta(z_q(yq))]

    vanishes at the truth for any measurable T.  With T(y) = log(1+y) the
    synthetic term is bounded by log(1 + exp(clamp)) * ||linpar||, so
    gradients stay finite even when the decoder has not yet converged.

    Sampling, log_prob, and eta are unchanged (identical to PoissonGLM).
    Only zq_log is affected via the overridden T().
    """

    def T(self, y):
        return torch.log1p(y.float())

    def __repr__(self):
        return f"PoissonLog1pGLM(linpar={self.linpar})"


class PoissonMixedTGLM(PoissonGLM):
    """
    Poisson GLM using T(y) = y + log(1+y) in the ZQE estimating equation.

    This is the equal-weight sum of the canonical (T=y) and the stable
    (T=log1p) estimating equations:

        [E(y·η) - E_θ(yq·η)]  +  [E(log1p(y)·η) - E_θ(log1p(yq)·η)]  = 0

    Because η does not depend on T, summing the two equations is equivalent
    to a single equation with T_mix(y) = y + log(1+y).  This recovers the
    gradient signal of the canonical score (y·η has the right Fisher
    efficiency) while log1p(y) regularises the centring term against
    large-count explosions.

    Sampling, log_prob, and eta are unchanged.
    """

    def T(self, y):
        y = y.float()
        return y + torch.log1p(y)

    def __repr__(self):
        return f"PoissonMixedTGLM(linpar={self.linpar})"


class PoissonSqrtGLM(PoissonGLM):
    """
    Poisson GLM using T(y) = sqrt(y) in the ZQE estimating equation.

    sqrt is the variance-stabilising transformation for Poisson (delta method:
    Var[sqrt(Y)] ≈ 1/4).  It compresses large counts more aggressively than
    log1p, giving a flatter signal across the dynamic range of scRNA-seq data.
    Unlike T=y it never causes gradient explosion at random decoder init.
    """

    def T(self, y):
        return torch.sqrt(y.float().clamp_min(0.0))

    def __repr__(self):
        return f"PoissonSqrtGLM(linpar={self.linpar})"


class PoissonLog1pSqrtGLM(PoissonGLM):
    """
    Poisson GLM using T(y) = (log1p(y) + sqrt(y)) / 2.

    Equal-weight average of the two best-performing transforms:
    log1p for low-count sensitivity and sqrt for variance stabilisation.
    """

    def T(self, y):
        y = y.float().clamp_min(0.0)
        return (torch.log1p(y) + torch.sqrt(y)) / 2.0

    def __repr__(self):
        return f"PoissonLog1pSqrtGLM(linpar={self.linpar})"


class PoissonMultiTGLM(PoissonGLM):
    """
    Poisson GLM using T(y) = (1/3)*[log1p(y) + sqrt(y) + y/(1+y)] as a
    uniform-weight mixture of three bounded/stable transformations.

    Motivation: each transformation captures different aspects of the count
    distribution:
      - log1p(y): emphasises low counts, log-compresses high counts
      - sqrt(y):  variance-stabilising, intermediate compression
      - y/(1+y):  bounded in [0,1), de-emphasises outlier counts

    All three are bounded/slow-growing so the centring term stays finite.
    Equal weights keep the mixture scale neutral.  The resulting estimating
    equation is the average of three individually valid ZQE equations.
    """

    def T(self, y):
        y = y.float().clamp_min(0.0)
        return (torch.log1p(y) + torch.sqrt(y) + y / (1.0 + y)) / 3.0

    def __repr__(self):
        return f"PoissonMultiTGLM(linpar={self.linpar})"


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

        mu = torch.exp(torch.clamp(linpar, max=10.0))
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
