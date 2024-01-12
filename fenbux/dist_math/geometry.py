from jax import numpy as jnp
from jaxtyping import Array


def geometric_logpmf(x, p) -> Array:
    return jnp.log(p) + (x - 1) * jnp.log1p(-p)


def geometric_pmf(x, p) -> Array:
    return jnp.exp(geometric_logpmf(x, p))


def geometric_cdf(x, p) -> Array:
    return 1 - jnp.power(1 - p, x)


def geometric_logcdf(x, p) -> Array:
    return jnp.log(geometric_cdf(x, p))


def geometric_sf(x, p) -> Array:
    return jnp.power(1 - p, x)


def geometric_logsf(x, p) -> Array:
    return jnp.log(geometric_sf(x, p))


def geometric_ppf(x, p) -> Array:
    return jnp.ceil(jnp.log(1 - x) / jnp.log1p(-p))
