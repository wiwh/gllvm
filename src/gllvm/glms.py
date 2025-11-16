from abc import ABC, abstractmethod
import torch
import torch.distributions as D
from typing import Any

__all__ = [
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
    """Abstract mixin for Generalized Linear Models.

    Each GLM defines:
      - linpar (eta): linear predictor
      - scale (phi): dispersion or scale parameter, always present internally
    """

    @abstractmethod
    def eta(self) -> torch.Tensor:
        """Canonical parameter eta."""
        pass

    @abstractmethod
    def T(self, y: torch.Tensor) -> torch.Tensor:
        """Sufficient statistic T(y)."""
        pass

    def zq_log(self, y: torch.Tensor) -> torch.Tensor:
        # TODO: check size of T(Y) first --- before taking sum(-1).

        """Informative part of the complete-data log-likelihood."""
        return (self.T(y) * self.eta() / self.scale).sum(-1)


# ============================================================
# Gaussian GLM
# ============================================================


class GaussianGLM(D.Normal, GLMMixin):
    """Gaussian GLM (identity link).

    Var[Y] = phi = scale
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

    @property
    def std(self) -> torch.Tensor:
        return torch.sqrt(self.scale)

    @property
    def variance(self) -> torch.Tensor:
        return self.scale


# ============================================================
# Poisson GLM
# ============================================================


class PoissonGLM(D.Poisson, GLMMixin):
    """Canonical Poisson GLM (log link).

    Var[Y] = mu, fixed scale = 1 (internally)
    """

    def __init__(self, linpar: torch.Tensor):
        self.linpar = linpar
        self.scale = torch.tensor(1.0, device=linpar.device)  # fixed, internal only
        rate = torch.exp(linpar)
        super().__init__(rate=rate)

    def eta(self) -> torch.Tensor:
        return self.linpar

    def T(self, y: torch.Tensor) -> torch.Tensor:
        return y


# ============================================================
# Gamma GLM
# ============================================================


class GammaGLM(D.Gamma, GLMMixin):
    """Gamma GLM with log link.

    Var[Y] = scale * mu^2, shape = 1 / scale
    """

    def __init__(self, linpar: torch.Tensor, scale: float | torch.Tensor = 1.0):
        self.linpar = linpar
        self.scale = torch.as_tensor(scale, device=linpar.device)
        mu = torch.exp(linpar)
        shape = 1.0 / self.scale
        rate = shape / mu
        super().__init__(concentration=shape, rate=rate)

    def eta(self) -> torch.Tensor:
        return self.linpar

    def T(self, y: torch.Tensor) -> torch.Tensor:
        return torch.log(y)

    @property
    def shape(self) -> torch.Tensor:
        """Shape (alpha) = 1 / scale."""
        return 1.0 / self.scale

    @property
    def rate_param(self) -> torch.Tensor:
        """Rate (beta) = alpha / mu."""
        mu = torch.exp(self.linpar)
        return self.shape / mu


# ============================================================
# Negative Binomial GLM
# ============================================================


class NegativeBinomialGLM(D.NegativeBinomial, GLMMixin):
    """Negative Binomial GLM (log link).

    Var[Y] = mu + scale * mu^2, total_count = 1 / scale
    """

    def __init__(self, linpar: torch.Tensor, scale: float | torch.Tensor = 1.0):
        self.linpar = linpar
        self.scale = torch.as_tensor(scale, device=linpar.device)
        mu = torch.exp(linpar)
        total_count = 1.0 / self.scale
        p = total_count / (total_count + mu)
        super().__init__(total_count=total_count, probs=p)

    def eta(self) -> torch.Tensor:
        return self.linpar

    def T(self, y: torch.Tensor) -> torch.Tensor:
        return y

    @property
    def total_count_param(self) -> torch.Tensor:
        return 1.0 / self.scale

    @property
    def probs_param(self) -> torch.Tensor:
        mu = torch.exp(self.linpar)
        r = self.total_count_param
        return r / (r + mu)


# ============================================================
# Binomial GLM
# ============================================================


class BinomialGLM(D.Binomial, GLMMixin):
    """Binomial GLM (logit link).

    Var[Y] = n * p * (1 - p), fixed scale = 1 (internally)
    """

    def __init__(self, linpar: torch.Tensor, total_count: Any = 1):
        self.linpar = linpar
        self.total_count = total_count
        self.scale = torch.tensor(1.0, device=linpar.device)  # internal only
        super().__init__(total_count=total_count, logits=linpar)

    def eta(self) -> torch.Tensor:
        return self.linpar

    def T(self, y: torch.Tensor) -> torch.Tensor:
        return y

    @property
    def probs_param(self) -> torch.Tensor:
        return torch.sigmoid(self.linpar)


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    linpar = torch.tensor([0.0, 1.0, -1.0])
    scale = torch.tensor([1, 0.5, 2])

    print("=== Gaussian GLM ===")
    g = GaussianGLM(linpar, scale=scale)
    y = g.sample()
    print("Samples:", y)
    print("scale:", g.scale)
    print("zq_log:", g.zq_log(y))
    print()

    print("=== Poisson GLM ===")
    p = PoissonGLM(linpar)
    y = p.sample()
    print("Samples:", y)
    print("zq_log:", p.zq_log(y))
    print()

    print("=== Gamma GLM ===")
    ga = GammaGLM(linpar, scale=scale)
    y = ga.sample()
    print("Samples:", y)
    print("shape:", ga.shape)
    print("rate:", ga.rate)
    print("zq_log:", ga.zq_log(y))
    print()

    print("=== Negative Binomial GLM ===")
    nb = NegativeBinomialGLM(linpar, scale=scale)
    y = nb.sample()
    print("Samples:", y)
    print("total_count:", nb.total_count_param)
    print("p:", nb.probs_param)
    print("zq_log:", nb.zq_log(y))
    print()

    print("=== Binomial GLM ===")
    b = BinomialGLM(linpar, total_count=5)
    y = b.sample()
    print("Samples:", y)
    print("p:", b.probs_param)
    print("zq_log:", b.zq_log(y))
    print()
