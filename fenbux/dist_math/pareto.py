from jax import numpy as jnp
from jax._src.numpy.util import promote_args_inexact
from jaxtyping import Array


def pareto_logpdf(x, shape, scale) -> Array:
    x, shape, scale = promote_args_inexact("pareto_logpdf", x, shape, scale)
    return jnp.log(shape) + jnp.log(scale) * shape - jnp.log(x) * (shape + 1.0)


def pareto_pdf(x, shape, scale) -> Array:
    x, shape, scale = promote_args_inexact("pareto_pdf", x, shape, scale)
    return jnp.exp(pareto_logpdf(x, shape, scale))


def pareto_cdf(x, shape, scale) -> Array:
    x, shape, scale = promote_args_inexact("pareto_cdf", x, shape, scale)
    return 1 - (scale / x) ** shape


def pareto_logcdf(x, shape, scale) -> Array:
    x, shape, scale = promote_args_inexact("pareto_logcdf", x, shape, scale)
    return jnp.log1p(-((scale / x) ** shape))


def pareto_sf(x, shape, scale) -> Array:
    x, shape, scale = promote_args_inexact("pareto_sf", x, shape, scale)
    return (scale / x) ** shape


def pareto_logsf(x, shape, scale) -> Array:
    x, shape, scale = promote_args_inexact("pareto_logsf", x, shape, scale)
    return jnp.log((scale / x)) * shape


def pareto_ppf(x, shape, scale) -> Array:
    x, shape, scale = promote_args_inexact("pareto_ppf", x, shape, scale)
    return scale / (1 - x) ** (1 / shape)


def pareto_isf(x, shape, scale) -> Array:
    x, shape, scale = promote_args_inexact("pareto_isf", x, shape, scale)
    return pareto_ppf(1 - x, shape, scale)
