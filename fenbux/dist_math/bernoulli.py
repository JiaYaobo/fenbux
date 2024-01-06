from jax import numpy as jnp
from jaxtyping import Array


def bernoulli_logpmf(x, p) -> Array:
    return jnp.where(x, jnp.log(p), jnp.log1p(-p))


def bernoulli_pmf(x, p) -> Array:
    return jnp.where(x, p, 1 - p)


def bernoulli_cdf(x, p) -> Array:
    return jnp.where(x, 1, 1 - p)


def bernoulli_logcdf(x, p) -> Array:
    return jnp.where(x, 0, jnp.log1p(-p))


def bernoulli_sf(x, p) -> Array:
    return jnp.where(x, 0, 1 - p)


def bernoulli_logsf(x, p) -> Array:
    return jnp.where(x, -jnp.inf, jnp.log(p))


def bernoulli_ppf(x, p) -> Array:
    return jnp.where(x > 1 - p, 1.0, 0.0)


def bernoulli_mgf(t, p) -> Array:
    return p * jnp.exp(t) + (1 - p)


def bernoulli_cf(t, p) -> Array:
    return p * jnp.exp(1j * t) + (1 - p)
