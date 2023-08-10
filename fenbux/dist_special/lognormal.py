from jax import numpy as jnp
from jax._src.numpy.util import promote_args_inexact
from jax.scipy.special import erf, erfinv
from jaxtyping import Array


def lognormal_logpdf(x, mean, sigma) -> Array:
    x, mean, sigma = promote_args_inexact("lognormal_logpdf", x, mean, sigma)
    return (
        -0.5 * jnp.log(2 * jnp.pi)
        - jnp.log(sigma)
        - jnp.log(x)
        - ((jnp.log(x) - mean) / sigma) ** 2 / 2
    )


def lognormal_pdf(x, mean, sigma) -> Array:
    x, mean, sigma = promote_args_inexact("lognormal_pdf", x, mean, sigma)
    return jnp.exp(lognormal_logpdf(x, mean, sigma))


def lognormal_cdf(x, mean, sigma) -> Array:
    x, mean, sigma = promote_args_inexact("lognormal_cdf", x, mean, sigma)
    return 0.5 * (1 + erf((jnp.log(x) - mean) / (sigma * jnp.sqrt(2))))


def lognormal_logcdf(x, mean, sigma) -> Array:
    x, mean, sigma = promote_args_inexact("lognormal_logcdf", x, mean, sigma)
    return jnp.log(0.5 * (1 + erf((jnp.log(x) - mean) / (sigma * jnp.sqrt(2)))))


def lognormal_sf(x, mean, sigma) -> Array:
    x, mean, sigma = promote_args_inexact("lognormal_sf", x, mean, sigma)
    return 1 - lognormal_cdf(x, mean, sigma)


def lognormal_logsf(x, mean, sigma) -> Array:
    x, mean, sigma = promote_args_inexact("lognormal_logsf", x, mean, sigma)
    return jnp.log(lognormal_sf(x, mean, sigma))


def lognormal_ppf(x, mean, sigma) -> Array:
    x, mean, sigma = promote_args_inexact("lognormal_ppf", x, mean, sigma)
    return jnp.exp(mean + sigma * erfinv(2 * x - 1) * jnp.sqrt(2))


def lognormal_isf(x, mean, sigma) -> Array:
    x, mean, sigma = promote_args_inexact("lognormal_isf", x, mean, sigma)
    return lognormal_ppf(1 - x, mean, sigma)
