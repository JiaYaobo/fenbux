from jax import numpy as jnp
from jaxtyping import Array


def laplace_logpdf(x, loc, scale) -> Array:
    return -jnp.log(2 * scale) - jnp.abs(x - loc) / scale


def laplace_pdf(x, loc, scale) -> Array:
    return jnp.exp(laplace_logpdf(x, loc, scale))


def laplace_cdf(x, loc, scale) -> Array:
    return 0.5 + 0.5 * jnp.sign(x - loc) * (1 - jnp.exp(-jnp.abs(x - loc) / scale))


def laplace_logcdf(x, loc, scale) -> Array:
    return jnp.log(laplace_cdf(x, loc, scale))


def laplace_sf(x, loc, scale) -> Array:
    return 1 - laplace_cdf(x, loc, scale)


def laplace_logsf(x, loc, scale) -> Array:
    return jnp.log(laplace_sf(x, loc, scale))


def laplace_ppf(x, loc, scale) -> Array:
    return loc - scale * jnp.sign(x - 0.5) * jnp.log1p(-2 * jnp.abs(x - 0.5))
