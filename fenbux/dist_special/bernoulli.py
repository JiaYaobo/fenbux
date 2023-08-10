from jax import numpy as jnp
from jax._src.numpy.util import promote_args_inexact
from jaxtyping import Array


def bernoulli_logpmf(x, p) -> Array:
    x, p = promote_args_inexact("bernoulli_logpmf", x, p)
    return jnp.where(x, jnp.log(p), jnp.log1p(-p))


def bernoulli_pmf(x, p) -> Array:
    x, p = promote_args_inexact("bernoulli_pmf", x, p)
    return jnp.where(x, p, 1 - p)


def bernoulli_cdf(x, p) -> Array:
    x, p = promote_args_inexact("bernoulli_cdf", x, p)
    return jnp.where(x, 1, 1 - p)


def bernoulli_logcdf(x, p) -> Array:
    x, p = promote_args_inexact("bernoulli_logcdf", x, p)
    return jnp.where(x, 0, jnp.log1p(-p))


def bernoulli_sf(x, p) -> Array:
    x, p = promote_args_inexact("bernoulli_sf", x, p)
    return jnp.where(x, 0, 1 - p)


def bernoulli_logsf(x, p) -> Array:
    x, p = promote_args_inexact("bernoulli_logsf", x, p)
    return jnp.where(x, -jnp.inf, jnp.log(p))


def bernoulli_ppf(x, p) -> Array:
    x, p = promote_args_inexact("bernoulli_ppf", x, p)
    return jnp.where(x > 1 - p, 1.0, 0.0)


def bernoulli_mgf(t, p) -> Array:
    t, p = promote_args_inexact("bernoulli_mgf", t, p)
    return p * jnp.exp(t) + (1 - p)


def bernoulli_cf(t, p) -> Array:
    t, p = promote_args_inexact("bernoulli_cf", t, p)
    return p * jnp.exp(1j * t) + (1 - p)
