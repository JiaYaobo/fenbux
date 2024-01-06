from jax import numpy as jnp
from jaxtyping import Array


def pareto_logpdf(x, shape, scale) -> Array:
    return jnp.log(shape) + jnp.log(scale) * shape - jnp.log(x) * (shape + 1.0)


def pareto_pdf(x, shape, scale) -> Array:
    return jnp.exp(pareto_logpdf(x, shape, scale))


def pareto_cdf(x, shape, scale) -> Array:
    return 1 - (scale / x) ** shape


def pareto_logcdf(x, shape, scale) -> Array:
    return jnp.log1p(-((scale / x) ** shape))


def pareto_sf(x, shape, scale) -> Array:
    return (scale / x) ** shape


def pareto_logsf(x, shape, scale) -> Array:
    return jnp.log((scale / x)) * shape


def pareto_ppf(x, shape, scale) -> Array:
    return scale / (1 - x) ** (1 / shape)


def pareto_isf(x, shape, scale) -> Array:
    return pareto_ppf(1 - x, shape, scale)
