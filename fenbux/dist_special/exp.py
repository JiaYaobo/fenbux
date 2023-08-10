from jax import numpy as jnp
from jax._src.numpy.util import promote_args_inexact
from jax.scipy.special import xlog1py
from jaxtyping import Array


def exp_logpdf(x, r) -> Array:
    x, r = promote_args_inexact("exp_logpdf", x, r)
    return jnp.log(r) - r * x


def exp_pdf(x, r) -> Array:
    x, r = promote_args_inexact("exp_pdf", x, r)
    return r * jnp.exp(-r * x)


def exp_cdf(x, r) -> Array:
    x, r = promote_args_inexact("exp_cdf", x, r)
    return 1.0 - jnp.exp(-r * x)


def exp_logcdf(x, r) -> Array:
    x, r = promote_args_inexact("exp_logcdf", x, r)
    return jnp.log1p(-jnp.exp(-r * x))


def exp_sf(x, r) -> Array:
    x, r = promote_args_inexact("exp_sf", x, r)
    return jnp.exp(-r * x)


def exp_logsf(x, r) -> Array:
    x, r = promote_args_inexact("exp_logsf", x, r)
    return xlog1py(-jnp.exp(-r * x), -1.0)


def exp_isf(x, r) -> Array:
    x, r = promote_args_inexact("exp_isf", x, r)
    return -jnp.log1p(-x) / r


def exp_ppf(q, r) -> Array:
    q, r = promote_args_inexact("exp_ppf", q, r)
    return -jnp.log1p(-q) / r


def exp_mgf(t, r) -> Array:
    t, r = promote_args_inexact("exp_mgf", t, r)
    return r / (r - t)


def exp_cf(t, r) -> Array:
    t, r = promote_args_inexact("exp_cf", t, r)
    return r / (r - 1j * t)
