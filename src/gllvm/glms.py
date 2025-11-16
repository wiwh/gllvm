import torch
import torch.distributions as D
from dataclasses import dataclass
from abc import ABC, abstractmethod

__all__ = [
    "BaseGLM",
    "GaussianGLM",
    "PoissonGLM",
    "GammaGLM",
    "NegativeBinomialGLM",
    "BinomialGLM",
]


@dataclass
class BaseGLM(ABC):
    """
    Abstract base class for canonical GLMs. Non-canonical GLMs (with non-canonical link functions) can be
    implemented by the user.

    Each derived GLM must define:
        - T(y):       sufficent statistic
        - mean(linpar):  mean parameter μ = g^{-1}(η)
        - make_dist(mean): a torch.distributions.* object
    """

    # Placeholder for future non-canonical link functions implementation.
    @abstractmethod
    def eta(self, linpar: torch.Tensor) -> torch.Tensor:
        # For canonical link functions, this is the identity function.
        pass

    # ----- Abstract Methods -----

    @abstractmethod
    def T(self, y: torch.Tensor) -> torch.Tensor:
        """The complete-data sufficient statistic. Can be a vector, depending on the model.
        It may differ than the sufficient statistic from the log-likelihood, with potential efficiency and computational implications.
        """
        pass

    @abstractmethod
    def mean(self, linpar: torch.Tensor) -> torch.Tensor:
        """
        Mean μ = g^{-1}(linpar).
        In canonical form, this is the inverse canonical link.
        """
        pass

    @abstractmethod
    def make_dist(self, linpar: torch.Tensor) -> D.Distribution:
        """
        Return a torch.distributions object parameterized in mean space.
        The exact way to do it often depends on the specific GLM being implemented and the pytorch implementation itself.
        It may or may not use a specific link function.
        """
        pass

    @abstractmethod
    def sample(self, linpar: torch.Tensor) -> torch.Tensor:
        """
        Sample from the distribution given the linear predictor.
        """
        pass

    # Get the log_prob from torch.distributions API
    def log_prob(self, y: torch.Tensor, linpar: torch.Tensor) -> torch.Tensor:
        """
        Standard complete-data log-likelihood (for reference).
        """
        mean = self.mean(linpar)
        return self.make_dist(mean).log_prob(y)

    # Get the data-part of the log_prob for the Zq estimator
    def zq_log(self, y: torch.Tensor, linpar: torch.Tensor) -> torch.Tensor:
        """
        The *informative* part of the complete-data log-likelihood,
        i.e. T(y)^T * eta(linpar).
        This is equal to self.log_prob - A(eta) - B(y) where:
            * A(eta) is the log-partition function, and
            * B(y) is the base measure.
        """
        eta = self.eta(linpar)
        T_y = self.T(y)
        return (T_y * eta).sum(-1)  # sum over features


# ==============================
# Canonical GLMs
# ==============================
#
# Canonical GLMs are special cases where
# eta(linpar) = linpar
#
# Users can extend the BaseGLM class and can easily implement their own non-canonical
# version, or other GLMs.


@dataclass
class GaussianGLM(BaseGLM):
    """Gaussian GLM (identity link).

    Args:
        std: Standard deviation (scalar or tensor, broadcastable to mean).
             Allows per-element noise scales.
    """

    std: float | torch.Tensor = 1.0  # scalar or tensor

    def eta(self, linpar: torch.Tensor) -> torch.Tensor:
        """Identity link: eta = linpar."""
        return linpar

    def T(self, y: torch.Tensor) -> torch.Tensor:
        """Sufficient statistic."""
        return y

    def mean(self, linpar: torch.Tensor) -> torch.Tensor:
        """Inverse link: identity."""
        return linpar

    def make_dist(self, linpar: torch.Tensor) -> D.Distribution:
        mu = self.mean(linpar)
        return D.Normal(mu, self.std)

    def sample(self, linpar: torch.Tensor) -> torch.Tensor:
        """Draw samples."""
        return self.make_dist(linpar).sample()


@dataclass
class PoissonGLM(BaseGLM):
    """Canonical Poisson GLM (log link)."""

    def eta(self, linpar: torch.Tensor) -> torch.Tensor:
        return linpar

    def T(self, y: torch.Tensor) -> torch.Tensor:
        return y

    def mean(self, linpar: torch.Tensor) -> torch.Tensor:
        return torch.exp(linpar)

    def make_dist(self, linpar: torch.Tensor) -> D.Distribution:
        rate = self.mean(linpar)
        return D.Poisson(rate)

    def sample(self, linpar: torch.Tensor) -> torch.Tensor:
        return self.make_dist(linpar).sample()


@dataclass
class GammaGLM(BaseGLM):
    """Gamma GLM with log link (non-canonical).

    Link: η = log(μ)
    Mean: μ = exp(η)
    Canonical parameter: θ = -1 / μ  (not used here)

    Args:
        a: Shape parameter (often noted alpha).
    Remarks:
        The rate parameter b (beta) is set to beta = alpha / mu, ensuring E[Y] = μ > 0.
    """

    a: float = 1.0  # shape parameter

    def eta(self, linpar: torch.Tensor) -> torch.Tensor:
        """Linear predictor (η = Xβ). For log link, η = log(μ)."""
        return linpar

    def T(self, y: torch.Tensor) -> torch.Tensor:
        """Sufficient statistic under log-link parameterization."""
        return torch.log(y)

    def mean(self, linpar: torch.Tensor) -> torch.Tensor:
        """Inverse link: μ = exp(η)."""
        return torch.exp(linpar)

    def make_dist(self, linpar: torch.Tensor) -> D.Distribution:
        """Return Gamma(a, b) with b = a / μ so that E[Y] = μ."""
        mu = self.mean(linpar)
        b = self.a / mu  # rate = a / μ
        return D.Gamma(concentration=self.a, rate=b)

    def sample(self, linpar: torch.Tensor) -> torch.Tensor:
        return self.make_dist(linpar).sample()


@dataclass
class NegativeBinomialGLM(BaseGLM):
    """Canonical Negative Binomial GLM (log link)."""

    def eta(self, linpar: torch.Tensor) -> torch.Tensor:
        return linpar

    def T(self, y: torch.Tensor) -> torch.Tensor:
        return y

    def mean(self, linpar: torch.Tensor) -> torch.Tensor:
        return torch.exp(linpar)

    def make_dist(self, linpar: torch.Tensor) -> D.Distribution:
        mean = self.mean(linpar)
        # fixed dispersion r = 1
        r = 1.0
        p = r / (r + mean)
        return D.NegativeBinomial(total_count=r, probs=p)

    def sample(self, linpar: torch.Tensor) -> torch.Tensor:
        return self.make_dist(linpar).sample()


@dataclass
class BinomialGLM(BaseGLM):
    """Canonical Binomial GLM (logit link).

    Args:
        n_trials: Number of trials for the Binomial distribution. Defaults to 1 (=Bernoulli).

    """

    n_trials: int = 1  # Number of trials for the Binomial distribution

    def eta(self, linpar: torch.Tensor) -> torch.Tensor:
        return linpar

    def T(self, y: torch.Tensor) -> torch.Tensor:
        return y

    def mean(self, linpar: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(linpar)  # inverse logit

    def make_dist(self, linpar: torch.Tensor) -> D.Distribution:
        probs = self.mean(linpar)
        return D.Binomial(total_count=self.n_trials, probs=probs)

    def sample(self, linpar: torch.Tensor) -> torch.Tensor:
        return self.make_dist(linpar).sample()


if __name__ == "__main__":
    """Example usage of canonical GLMs."""

    linpar = torch.tensor([0.0, 1.0, -1.0])

    # --- Binomial ---
    binom = BinomialGLM(n_trials=5)
    print("=== Binomial GLM ===")
    print("Samples:", binom.sample(linpar))
    print("Means:", binom.mean(linpar))
    print()

    # --- Poisson ---
    pois = PoissonGLM()
    print("=== Poisson GLM ===")
    print("Samples:", pois.sample(linpar))
    print("Means:", pois.mean(linpar))
    print()

    # --- Gaussian ---
    gauss = GaussianGLM(std=torch.tensor([2.0]))
    linpar_g = torch.tensor([0.0, 1.0, -0.5])
    samples_g = gauss.sample(linpar_g)
    print("=== Gaussian GLM ===")
    print("Samples:", samples_g)
    print("Means:", gauss.mean(linpar_g))
    print("log_prob:", gauss.log_prob(samples_g, linpar_g))
    print("zq_log:", gauss.zq_log(samples_g, linpar_g))
    print()

    # --- Gamma ---
    gamma = GammaGLM(a=2.0)
    linpar_ga = torch.tensor([-1.0, -0.5, -2.0])  # ensure μ > 0
    samples_ga = gamma.sample(linpar_ga)
    print("=== Gamma GLM ===")
    print("Samples:", samples_ga)
    print("Means:", gamma.mean(linpar_ga))
    print("log_prob:", gamma.log_prob(samples_ga, linpar_ga))
    print("zq_log:", gamma.zq_log(samples_ga, linpar_ga))
    print()

    # --- Negative Binomial ---
    nb = NegativeBinomialGLM()
    samples_nb = nb.sample(linpar)
    print("=== Negative Binomial GLM ===")
    print("Samples:", samples_nb)
    print("Means:", nb.mean(linpar))
    print("log_prob:", nb.log_prob(samples_nb, linpar))
    print("zq_log:", nb.zq_log(samples_nb, linpar))
    print()
