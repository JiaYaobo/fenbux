from jax import numpy as jnp
from jax._src.numpy.util import promote_args_inexact
from jax.scipy.special import ndtr, ndtri
from jaxtyping import Array


def normal_logpdf(x, mean, sigma) -> Array:
    x, mean, sigma = promote_args_inexact("normal_logpdf", x, mean, sigma)
    return -0.5 * jnp.log(2 * jnp.pi) - jnp.log(sigma) - ((x - mean) / sigma) ** 2 / 2


def normal_pdf(x, mean, sigma) -> Array:
    x, mean, sigma = promote_args_inexact("normal_pdf", x, mean, sigma)
    return jnp.exp(normal_logpdf(x, mean, sigma))


def normal_cdf(x, mean, sigma) -> Array:
    x, mean, sigma = promote_args_inexact("normal_cdf", x, mean, sigma)
    return ndtr((x - mean) / sigma)


def normal_logcdf(x, mean, sigma) -> Array:
    x, mean, sigma = promote_args_inexact("normal_logcdf", x, mean, sigma)
    return jnp.log(normal_cdf(x, mean, sigma))


def normal_sf(x, mean, sigma) -> Array:
    x, mean, sigma = promote_args_inexact("normal_sf", x, mean, sigma)
    return 1 - normal_cdf(x, mean, sigma)


def normal_logsf(x, mean, sigma) -> Array:
    x, mean, sigma = promote_args_inexact("normal_logsf", x, mean, sigma)
    return jnp.log(normal_sf(x, mean, sigma))


def normal_ppf(x, mean, sigma) -> Array:
    x, mean, sigma = promote_args_inexact("normal_ppf", x, mean, sigma)
    return ndtri(x) * sigma + mean


def normal_isf(x, mean, sigma) -> Array:
    x, mean, sigma = promote_args_inexact("normal_isf", x, mean, sigma)
    return normal_ppf(1 - x, mean, sigma)


def normal_mgf(t, mean, sigma) -> Array:
    t, mean, sigma = promote_args_inexact("normal_mgf", t, mean, sigma)
    return jnp.exp(mean * t + sigma**2 * t**2 / 2)


def normal_cf(t, mean, sigma) -> Array:
    t, mean, sigma = promote_args_inexact("normal_cf", t, mean, sigma)
    return jnp.exp(1j * mean * t - sigma**2 * t**2 / 2)
