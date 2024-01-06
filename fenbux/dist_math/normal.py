from jax import numpy as jnp
from jax.scipy.special import ndtr, ndtri
from jaxtyping import Array


def normal_logpdf(x, mean=0, sigma=1) -> Array:
    return -0.5 * jnp.log(2 * jnp.pi) - jnp.log(sigma) - ((x - mean) / sigma) ** 2 / 2


def normal_pdf(x, mean=0, sigma=1) -> Array:
    return jnp.exp(normal_logpdf(x, mean, sigma))


def normal_cdf(x, mean=0, sigma=1) -> Array:
    return ndtr((x - mean) / sigma)


def normal_logcdf(x, mean=0, sigma=1) -> Array:
    return jnp.log(normal_cdf(x, mean, sigma))


def normal_sf(x, mean=0, sigma=1) -> Array:
    return 1 - normal_cdf(x, mean, sigma)


def normal_logsf(x, mean=0, sigma=1) -> Array:
    return jnp.log(normal_sf(x, mean, sigma))


def normal_ppf(x, mean=0, sigma=1) -> Array:
    return ndtri(x) * sigma + mean


def normal_isf(x, mean=0, sigma=1) -> Array:
    return normal_ppf(1 - x, mean, sigma)


def normal_mgf(t, mean=0, sigma=1) -> Array:
    return jnp.exp(mean * t + sigma**2 * t**2 / 2)


def normal_cf(t, mean=0, sigma=1) -> Array:
    return jnp.exp(1j * mean * t - sigma**2 * t**2 / 2)
