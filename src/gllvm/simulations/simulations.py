# simulations.py
import torch
from gllvm import GLLVM, GaussianGLM, PoissonGLM, BinomialGLM

__all__ = [
    "make_gaussian_only",
    "make_poisson_only",
    "make_binomial_only",
    "make_mixed",
    "make_sparse",
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


def _apply_latent_sparsity(g, zero_latent_cols: int):
    """
    Zero out `zero_latent_cols` columns of wz (i.e. latent dimensions that
    carry no signal for any response).  These dimensions become structural
    zeros in the loadings matrix — the model is rank-deficient by design.

    The columns to silence are chosen as the *last* `zero_latent_cols`
    latent dimensions so the behaviour is deterministic given the model size.

    Parameters
    ----------
    g : GLLVM
    zero_latent_cols : int
        Number of latent dimensions to silence (must be < g.q).
    """
    if zero_latent_cols <= 0:
        return
    if zero_latent_cols >= g.q:
        raise ValueError(
            f"zero_latent_cols={zero_latent_cols} must be < n_latent={g.q}. "
            "Silencing all latent dims leaves no signal."
        )
    mask = torch.ones(g.p, g.q)
    mask[:, -zero_latent_cols:] = 0.0  # last cols → structural zeros
    g.set_wz_mask(mask)
    with torch.no_grad():
        g.wz.data[:, -zero_latent_cols:] = 0.0  # also zero the weights themselves


def _apply_loading_sparsity(
    g, responses_per_latent: int, active_cols: int | None = None
):
    """
    For each active latent column, zero out all but `responses_per_latent`
    randomly chosen response rows.  Columns beyond `active_cols` are left as-is
    (they may already be zeroed by _apply_latent_sparsity).

    The random assignment uses a fixed seed so results are reproducible but
    differ per column — adjacent latents will naturally share some features
    (overlap), giving the model something to learn jointly.

    Parameters
    ----------
    g : GLLVM
    responses_per_latent : int
        Number of response features each active latent column connects to.
        Must be < g.p.  Choose < g.p to get sparse loadings; choose overlapping
        values (e.g. g.p // 3 with 5 latents) so latents share some responses.
    active_cols : int or None
        How many of the first columns are "active" (rest already zeroed).
        Defaults to g.q.
    """
    if responses_per_latent <= 0 or responses_per_latent >= g.p:
        return
    if active_cols is None:
        active_cols = g.q
    assert isinstance(active_cols, int)

    # Build a (p, q) mask; start from the existing mask (may have zeroed cols)
    try:
        mask = g.wz_mask.clone()  # (p, q) if mask already set
    except AttributeError:
        mask = torch.ones(g.p, g.q)

    rng = torch.Generator()
    rng.manual_seed(0)
    for k in range(active_cols):
        perm = torch.randperm(g.p, generator=rng)
        on = perm[:responses_per_latent]
        off = perm[responses_per_latent:]
        mask[on, k] = 1.0
        mask[off, k] = 0.0

    g.set_wz_mask(mask)
    with torch.no_grad():
        g.wz.data *= mask.to(g.wz.device)


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
        BinomialGLM,
        idx=list(range(n_response)),
        name="Binomial",
        params={"total_count": n_trials},
    )

    _random_init(g)
    return g


def make_mixed(
    n_latent,
    gaussian=0,
    poisson=0,
    binomial=0,
    binom_trials=10,
    zero_latent_cols: int = 0,
    wz_scale: float = 1.0,
    responses_per_latent: int | None = None,
    lower_tri=False,
):
    """
    Build a mixed-outcome GLLVM.

    Parameters
    ----------
    n_latent : int
        Latent dimension q.
    gaussian, poisson, binomial : int
        Number of responses of each GLM family.
    binom_trials : int
        total_count for BinomialGLM.
    zero_latent_cols : int, default 0
        Number of latent dimensions to silence via structural zeros in wz
        (the last `zero_latent_cols` columns become all-zero).  This creates
        a rank-deficient loadings matrix, which breaks the VAE encoder but
        does not affect the ZQE decoder under centering.
        Must be strictly less than n_latent.
    wz_scale : float, default 1.0
        Multiply the non-zero wz entries by this after random initialisation.
        Use values like 0.3 to keep Poisson rates in a reasonable range.

    Example
    -------
        # 10 latent dims, 3 are silent → encoder sees 10 dims but only 7 carry signal
        make_mixed(10, poisson=100, zero_latent_cols=3, wz_scale=0.3)
        # each active latent affects only 20 out of 100 responses (overlapping)
        make_mixed(10, poisson=100, zero_latent_cols=3,
                   responses_per_latent=20, wz_scale=0.3)
    """
    p = gaussian + poisson + binomial
    g = GLLVM(latent_dim=n_latent, output_dim=p, lower_tri=lower_tri)

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

    if wz_scale != 1.0:
        with torch.no_grad():
            g.wz.data *= wz_scale

    if zero_latent_cols > 0:
        _apply_latent_sparsity(g, zero_latent_cols)

    if responses_per_latent is not None:
        active_cols = n_latent - zero_latent_cols
        _apply_loading_sparsity(g, responses_per_latent, active_cols=active_cols)

    return g


def make_sparse(
    n_latent: int,
    poisson: int = 0,
    gaussian: int = 0,
    binomial: int = 0,
    binom_trials: int = 10,
    active_latent: int | None = None,
    wz_scale: float = 0.3,
    responses_per_latent: int | None = None,
    lower_tri=False,
):
    """
    Convenience wrapper for the sparse-loadings simulation setting.

    Creates a GLLVM where only `active_latent` latent dimensions carry signal;
    the remaining `n_latent - active_latent` dimensions are structural zeros.
    This deliberately breaks the VAE encoder (it wastes capacity on silent
    dims) while leaving the ZQE decoder consistent by design.

    Parameters
    ----------
    n_latent : int
        Total latent dimension q (including silent ones).
    active_latent : int or None
        Number of latent dims that actually carry signal.  Defaults to
        ceil(n_latent / 2), so roughly half the dims are silent.
    wz_scale : float, default 0.3
        Scale factor applied to wz after init (keeps Poisson rates sane).
    responses_per_latent : int or None
        If given, each active latent column connects to only this many response
        features (randomly chosen, with overlap between latents).  None = dense
        (every active latent affects all p responses, the previous default).

    Example
    -------
        g = make_sparse(n_latent=10, poisson=100, active_latent=5, wz_scale=0.3)
        # g.wz[:, 5:] == 0  →  latent dims 5-9 are structurally silent

        g = make_sparse(n_latent=10, poisson=100, active_latent=5,
                        responses_per_latent=30, wz_scale=0.3)
        # each active latent affects 30/100 features (overlapping blocks)
    """
    if active_latent is None:
        import math

        active_latent = math.ceil(n_latent / 2)

    zero_latent_cols = n_latent - active_latent
    return make_mixed(
        n_latent=n_latent,
        poisson=poisson,
        gaussian=gaussian,
        binomial=binomial,
        binom_trials=binom_trials,
        zero_latent_cols=zero_latent_cols,
        wz_scale=wz_scale,
        responses_per_latent=responses_per_latent,
        lower_tri=lower_tri,
    )


# ---------------------------------------------
# main simulation API
# ---------------------------------------------
def simulate(gllvm, n_samples, device="cpu"):
    gllvm = gllvm.to(device)

    with torch.no_grad():
        z = gllvm.sample_z(n_samples)
        y = gllvm.sample(z=z)
    return y, z
