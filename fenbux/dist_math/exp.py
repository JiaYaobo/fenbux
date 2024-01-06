from jax import numpy as jnp
from jax.scipy.special import xlog1py
from jaxtyping import Array


def exp_logpdf(x, r) -> Array:
    return jnp.log(r) - r * x


def exp_pdf(x, r) -> Array:
    return r * jnp.exp(-r * x)


def exp_cdf(x, r) -> Array:
    return 1.0 - jnp.exp(-r * x)


def exp_logcdf(x, r) -> Array:
    return jnp.log1p(-jnp.exp(-r * x))


def exp_sf(x, r) -> Array:
    return jnp.exp(-r * x)


def exp_logsf(x, r) -> Array:
    return xlog1py(-jnp.exp(-r * x), -1.0)


def exp_isf(x, r) -> Array:
    return -jnp.log1p(-x) / r


def exp_ppf(q, r) -> Array:
    return -jnp.log1p(-q) / r


def exp_mgf(t, r) -> Array:
    return r / (r - t)


def exp_cf(t, r) -> Array:
    return r / (r - 1j * t)
