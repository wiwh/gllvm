"""
ZQ-GLM wrappers over PyTorch distributions.

Each GLM defines:
    - eta() : natural parameter (GLM linear predictor)
    - T(y)  : ZQE statistic (the "sufficient statistic" of the estimating
              equation; not strictly sufficient).  Defaults to the family's
              canonical statistic but is fully configurable per instance via
              the ``T=`` constructor argument, e.g.
                  PoissonGLM(linpar, T=torch.log1p)
              The override is threaded through GLMFamily.params, so it persists
              across the transient re-instantiations done by GLLVM.forward.
    - deviance(y) : classical GLM deviance
    - __init__: converts (linpar, scale) into parameters required by
                torch.distributions

Each GLM inherits from the corresponding torch.distributions class but exposes
a clean GLM interface suitable for GLLVMs and ZQ estimators.

Note on the two roles of T
--------------------------
The *decoder* T(y) configured here is the statistic in the ZQE estimating
equation E[T(y)·eta] - E_theta[T(Yq)·eta] = 0.  By the score-function identity
this can be ANY measurable map; it is decoupled from the generative model
(sampling / log_prob are unchanged).  The parameter-free encoders apply their
own internal linearising transform (log1p) as part of a Gaussian *proxy* model
for z|y — that is a separate concern from the decoder's T.
"""

from abc import ABC, abstractmethod
import torch
import torch.distributions as D
from typing import Any, Callable

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

    # Per-instance ZQE-statistic override.  None -> use the family canonical
    # statistic (_T_canonical).  Set via the ``T=`` constructor argument.
    _T_override: Callable[[torch.Tensor], torch.Tensor] | None = None

    @abstractmethod
    def eta(self) -> torch.Tensor:
        """Natural parameter eta."""
        pass

    @abstractmethod
    def _T_canonical(self, y: torch.Tensor) -> torch.Tensor:
        """Family canonical ZQE statistic (e.g. y for Poisson, log(y) for Gamma)."""
        pass

    def T(self, y: torch.Tensor) -> torch.Tensor:
        """
        ZQE statistic T(y).

        Returns the configured override if one was passed to ``__init__``
        (e.g. ``PoissonGLM(linpar, T=torch.log1p)``), otherwise the family
        canonical statistic.  Any measurable map is valid: the score-function
        identity makes the ZQE centring vanish at the truth for any T.
        """
        if self._T_override is not None:
            return self._T_override(y)
        return self._T_canonical(y)

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

    def __init__(self, linpar: torch.Tensor, scale: float | torch.Tensor = 1.0,
                 T: Callable[[torch.Tensor], torch.Tensor] | None = None):
        self.linpar = linpar
        self.scale = torch.as_tensor(scale, device=linpar.device)
        self._T_override = T
        std = torch.sqrt(self.scale)
        super().__init__(loc=linpar, scale=std)

    def eta(self):
        return self.linpar

    def _T_canonical(self, y):
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

    def __init__(self, linpar: torch.Tensor, scale: float | torch.Tensor = 1.0,
                 T: Callable[[torch.Tensor], torch.Tensor] | None = None):
        self.linpar = linpar
        self.scale = torch.tensor(1.0, device=linpar.device)
        self._T_override = T
        rate = torch.exp(linpar)
        super().__init__(rate=rate)

    def eta(self):
        return self.linpar

    def _T_canonical(self, y):
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
        t = "" if self._T_override is None else f", T={getattr(self._T_override, '__name__', 'custom')}"
        return f"PoissonGLM(linpar={self.linpar}{t})"


# ============================================================
# Gamma GLM
# ============================================================


class GammaGLM(D.Gamma, GLMMixin):
    """
    Gamma GLM with log link.
    Var[Y] = scale * mu^2
    scale = dispersion parameter
    """

    def __init__(self, linpar: torch.Tensor, scale: float | torch.Tensor = 1.0,
                 T: Callable[[torch.Tensor], torch.Tensor] | None = None):
        self.linpar = linpar
        self.scale = torch.as_tensor(scale, device=linpar.device)
        self._T_override = T

        mu = torch.exp(torch.clamp(linpar, max=10.0))
        concentration = 1.0 / self.scale  # shape = alpha
        rate = concentration / mu  # beta = alpha / mu

        super().__init__(concentration=concentration, rate=rate)

    def eta(self):
        return self.linpar

    def _T_canonical(self, y):
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

    def __init__(self, linpar: torch.Tensor, scale: float | torch.Tensor = 1.0,
                 T: Callable[[torch.Tensor], torch.Tensor] | None = None):
        self.linpar = linpar
        self.scale = torch.as_tensor(scale, device=linpar.device)
        self._T_override = T

        mu = torch.exp(linpar)
        theta = self.scale.clamp_min(1e-8)
        total_count = 1.0 / theta
        probs = total_count / (total_count + mu)

        super().__init__(total_count=total_count, probs=probs)

    def eta(self):
        return self.linpar

    def _T_canonical(self, y):
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
        T: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        self.linpar = linpar
        self.total_count = torch.as_tensor(total_count, device=linpar.device)
        self.scale = torch.tensor(1.0, device=linpar.device)  # unused
        self._T_override = T
        super().__init__(total_count=self.total_count, logits=linpar)

    def eta(self):
        return self.linpar

    def _T_canonical(self, y):
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
