from jax import numpy as jnp
from jax._src.numpy.util import promote_args_inexact
from jax.scipy.special import betainc, betaln, xlog1py, xlogy
from jaxtyping import Array
from tensorflow_probability.substrates.jax.math import betaincinv


def beta_logpdf(x, a, b) -> Array:
    x, a, b = promote_args_inexact("beta_logpdf", x, a, b)
    lp = xlog1py(b - 1.0, -x) + xlogy(a - 1.0, x)
    lp -= betaln(a, b)
    return lp


def beta_pdf(x, a, b) -> Array:
    x, a, b = promote_args_inexact("beta_pdf", x, a, b)
    return jnp.exp(beta_logpdf(x, a, b))


def beta_logcdf(x, a, b) -> Array:
    x, a, b = promote_args_inexact("beta_logcdf", x, a, b)
    return jnp.log(betainc(a, b, x))


def beta_cdf(x, a, b) -> Array:
    x, a, b = promote_args_inexact("beta_cdf", x, a, b)
    return betainc(a, b, x)


def beta_ppf(p, a, b) -> Array:
    p, a, b = promote_args_inexact("beta_quantile", p, a, b)
    return betaincinv(a, b, p)


def beta_sf(x, a, b) -> Array:
    x, a, b = promote_args_inexact("beta_sf", x, a, b)
    return betainc(a, b, 1 - x)


def beta_isf(x, a, b) -> Array:
    x, a, b = promote_args_inexact("beta_isf", x, a, b)
    return betaincinv(a, b, 1 - x)


def beta_logsf(x, a, b) -> Array:
    x, a, b = promote_args_inexact("beta_logsf", x, a, b)
    return jnp.log(betainc(a, b, 1 - x))
