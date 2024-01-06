from jax import numpy as jnp
from jax.scipy.special import betainc, betaln, xlog1py, xlogy
from jaxtyping import Array
from tensorflow_probability.substrates.jax.math import betaincinv


def beta_logpdf(x, a, b) -> Array:
    lp = xlog1py(b - 1.0, -x) + xlogy(a - 1.0, x)
    lp -= betaln(a, b)
    return lp


def beta_pdf(x, a, b) -> Array:
    return jnp.exp(beta_logpdf(x, a, b))


def beta_logcdf(x, a, b) -> Array:
    return jnp.log(betainc(a, b, x))


def beta_cdf(x, a, b) -> Array:
    return betainc(a, b, x)


def beta_ppf(p, a, b) -> Array:
    return betaincinv(a, b, p)


def beta_sf(x, a, b) -> Array:
    return betainc(a, b, 1 - x)


def beta_isf(x, a, b) -> Array:
    return betaincinv(a, b, 1 - x)


def beta_logsf(x, a, b) -> Array:
    return jnp.log(betainc(a, b, 1 - x))
