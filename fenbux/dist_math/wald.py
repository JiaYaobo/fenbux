from jax import numpy as jnp
from jax._src.numpy.util import promote_args_inexact
from jaxtyping import Array

from .normal import normal_cdf


def wald_logpdf(x, mu, lam) -> Array:
    x, mu, lam = promote_args_inexact("wald_logpdf", x, mu, lam)
    return jnp.log(
        lam
        / (2 * jnp.pi * x**3) ** 0.5
        * jnp.exp(-lam * (x - mu) ** 2 / (2 * mu**2 * x))
    )


def wald_pdf(x, mu, lam) -> Array:
    x, mu, lam = promote_args_inexact("wald_pdf", x, mu, lam)
    return jnp.exp(wald_logpdf(x, mu, lam))


def wald_cdf(x, mu, lam) -> Array:
    x, mu, lam = promote_args_inexact("wald_cdf", x, mu, lam)
    return normal_cdf(
        jnp.log(x / mu) / (lam**0.5 * (jnp.log(x / mu) + 1 / (2 * x)) ** 0.5)
    )


def wald_logcdf(x, mu, lam) -> Array:
    x, mu, lam = promote_args_inexact("wald_logcdf", x, mu, lam)
    return jnp.log(wald_cdf(x, mu, lam))


def wald_sf(x, mu, lam) -> Array:
    x, mu, lam = promote_args_inexact("wald_sf", x, mu, lam)
    return 1 - wald_cdf(x, mu, lam)


def wald_logsf(x, mu, lam) -> Array:
    x, mu, lam = promote_args_inexact("wald_logsf", x, mu, lam)
    return jnp.log(wald_sf(x, mu, lam))


def wald_ppf(x, mu, lam) -> Array:
    x, mu, lam = promote_args_inexact("wald_ppf", x, mu, lam)
    return mu * (1 + lam * (jnp.pi * x) ** 2 / 8) ** 2


def wald_isf(x, mu, lam) -> Array:
    x, mu, lam = promote_args_inexact("wald_isf", x, mu, lam)
    return wald_ppf(1 - x, mu, lam)
