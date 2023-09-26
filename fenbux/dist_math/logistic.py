from jax import numpy as jnp
from jax._src.numpy.util import promote_args_inexact
from jaxtyping import Array


def logistic_logpdf(x, loc, scale) -> Array:
    x, loc, scale = promote_args_inexact("logistic_logpdf", x, loc, scale)
    z = (x - loc) / scale
    return -jnp.log(scale) - z - 2 * jnp.log1p(jnp.exp(-z))


def logistic_pdf(x, loc, scale) -> Array:
    x, loc, scale = promote_args_inexact("logistic_pdf", x, loc, scale)
    return jnp.exp(logistic_logpdf(x, loc, scale))


def logistic_cdf(x, loc, scale) -> Array:
    x, loc, scale = promote_args_inexact("logistic_cdf", x, loc, scale)
    z = (x - loc) / scale
    return 1 / (1 + jnp.exp(-z))


def logistic_logcdf(x, loc, scale) -> Array:
    x, loc, scale = promote_args_inexact("logistic_logcdf", x, loc, scale)
    return -jnp.log1p(jnp.exp(-(x - loc) / scale))


def logistic_sf(x, loc, scale) -> Array:
    x, loc, scale = promote_args_inexact("logistic_sf", x, loc, scale)
    return 1 - logistic_cdf(x, loc, scale)


def logistic_logsf(x, loc, scale) -> Array:
    x, loc, scale = promote_args_inexact("logistic_logsf", x, loc, scale)
    return -jnp.log(logistic_cdf(x, loc, scale))


def logistic_ppf(x, loc, scale) -> Array:
    x, loc, scale = promote_args_inexact("logistic_ppf", x, loc, scale)
    return loc + scale * jnp.log(x / (1 - x))


def logistic_isf(x, loc, scale) -> Array:
    x, loc, scale = promote_args_inexact("logistic_isf", x, loc, scale)
    return logistic_ppf(1 - x, loc, scale)


def logistic_cf(t, loc, scale) -> Array:
    t, loc, scale = promote_args_inexact("logistic_cf", t, loc, scale)
    return jnp.exp(1j * t * loc) * (jnp.pi * scale * t) / jnp.sinh(jnp.pi * scale * t)
