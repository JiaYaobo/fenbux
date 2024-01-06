from jax import numpy as jnp
from jaxtyping import Array


def logistic_logpdf(x, loc, scale) -> Array:
    z = (x - loc) / scale
    return -jnp.log(scale) - z - 2 * jnp.log1p(jnp.exp(-z))


def logistic_pdf(x, loc, scale) -> Array:
    return jnp.exp(logistic_logpdf(x, loc, scale))


def logistic_cdf(x, loc, scale) -> Array:
    z = (x - loc) / scale
    return 1 / (1 + jnp.exp(-z))


def logistic_logcdf(x, loc, scale) -> Array:
    return -jnp.log1p(jnp.exp(-(x - loc) / scale))


def logistic_sf(x, loc, scale) -> Array:
    return 1 - logistic_cdf(x, loc, scale)


def logistic_logsf(x, loc, scale) -> Array:
    return -jnp.log(logistic_cdf(x, loc, scale))


def logistic_ppf(x, loc, scale) -> Array:
    return loc + scale * jnp.log(x / (1 - x))


def logistic_isf(x, loc, scale) -> Array:
    return logistic_ppf(1 - x, loc, scale)


def logistic_cf(t, loc, scale) -> Array:
    return jnp.exp(1j * t * loc) * (jnp.pi * scale * t) / jnp.sinh(jnp.pi * scale * t)
