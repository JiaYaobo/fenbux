import math

from jax import numpy as jnp
from jax.scipy.special import ndtr, ndtri
from jaxtyping import Array


_half_log2pi = 0.5 * math.log(2 * math.pi)


def normal_logpdf(x, mean=0, sigma=1) -> Array:
    log_unnormalized = -0.5 * jnp.square((x - mean) / sigma)
    log_normalization = _half_log2pi + jnp.log(sigma)
    return log_unnormalized - log_normalization


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
