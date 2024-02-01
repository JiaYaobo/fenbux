from jax import numpy as jnp
from jaxtyping import Array


def cauchy_logpdf(x, loc, scale) -> Array:
    log_unnormalized_prob = -jnp.log1p(((x - loc) / scale) ** 2)
    log_normalization = jnp.log(jnp.pi * scale)
    return log_unnormalized_prob - log_normalization


def cauchy_pdf(x, loc, scale) -> Array:
    return jnp.exp(cauchy_logpdf(x, loc, scale))


def cauchy_cdf(x, loc, scale) -> Array:
    return jnp.arctan((x - loc) / scale) / jnp.pi + 0.5


def cauchy_logcdf(x, loc, scale) -> Array:
    return jnp.log(cauchy_cdf(x, loc, scale))


def cauchy_ppf(p, loc, scale) -> Array:
    return loc + scale * jnp.tan(jnp.pi * (p - 0.5))


def cauchy_sf(x, loc, scale) -> Array:
    return 1 - cauchy_cdf(x, loc, scale)


def cauchy_logsf(x, loc, scale) -> Array:
    return jnp.log(cauchy_sf(x, loc, scale))


def cauchy_isf(x, loc, scale) -> Array:
    return cauchy_ppf(1 - x, loc, scale)
    
