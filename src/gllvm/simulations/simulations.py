# simulations.py
import torch
from gllvm import GLLVM, GaussianGLM, PoissonGLM, BinomialGLM

__all__ = [
    "make_gaussian_only",
    "make_poisson_only",
    "make_binomial_only",
    "make_mixed",
    "simulate",
]


# ---------------------------------------------
# helpers
# ---------------------------------------------
def _random_init(m, a=1.0):
    for name, p in m.named_parameters():
        if name == "log_scale":
            continue
        if p.ndim > 1:
            torch.nn.init.normal_(p, 0, a)
        else:
            torch.nn.init.normal_(p, 0, a)


# ---------------------------------------------
# simulation builders (presets)
# ---------------------------------------------
def make_gaussian_only(n_latent, n_response, scale=1.0):
    g = GLLVM(latent_dim=n_latent, output_dim=n_response)
    g.add_glm(GaussianGLM, idx=list(range(n_response)), name="Gaussian")

    _random_init(g)
    g.log_scale.data[:] = torch.log(torch.full_like(g.log_scale, scale))
    return g


def make_poisson_only(n_latent, n_response):
    g = GLLVM(latent_dim=n_latent, output_dim=n_response)
    g.add_glm(PoissonGLM, idx=list(range(n_response)), name="Poisson")

    _random_init(g)
    return g


def make_binomial_only(n_latent, n_response, n_trials=10):
    g = GLLVM(latent_dim=n_latent, output_dim=n_response)
    g.add_glm(
        BinomialGLM, idx=list(range(n_response)), name="Binomial", n_trials=n_trials
    )

    _random_init(g)
    return g


def make_mixed(n_latent, gaussian=0, poisson=0, binomial=0, binom_trials=10):
    """
    Example:
        make_mixed(2, gaussian=5, poisson=3, binomial=1)
    """
    p = gaussian + poisson + binomial
    g = GLLVM(latent_dim=n_latent, output_dim=p)

    idx = 0

    if gaussian > 0:
        g.add_glm(GaussianGLM, idx=list(range(idx, idx + gaussian)), name="G")
        idx += gaussian

    if poisson > 0:
        g.add_glm(PoissonGLM, idx=list(range(idx, idx + poisson)), name="P")
        idx += poisson

    if binomial > 0:
        g.add_glm(
            BinomialGLM,
            idx=list(range(idx, idx + binomial)),
            name="B",
            params={"total_count": binom_trials},
        )
        idx += binomial

    _random_init(g)
    return g


# ---------------------------------------------
# main simulation API
# ---------------------------------------------
def simulate(gllvm, n_samples, device="cpu"):
    gllvm = gllvm.to(device)

    with torch.no_grad():
        z = gllvm.sample_z(n_samples)
        y = gllvm.sample(z=z)
    return y, z
