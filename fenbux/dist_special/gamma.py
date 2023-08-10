from jax import numpy as jnp
from jax._src.numpy.util import promote_args_inexact
from jax.scipy.special import gammainc
from jax.scipy.stats.gamma import logpdf as _jax_gamma_logpdf
from jaxtyping import Array
from tensorflow_probability.substrates.jax.math import igammainv


def gamma_logpdf(x, shape, rate) -> Array:
    x, a, rate = promote_args_inexact("gamma_logpdf", x, shape, rate)
    return _jax_gamma_logpdf(x, a, scale=1 / rate)


def gamma_pdf(x, shape, rate) -> Array:
    x, a, rate = promote_args_inexact("gamma_pdf", x, shape, rate)
    return jnp.exp(gamma_logpdf(x, a, rate))


def gamma_cdf(x, shape, rate) -> Array:
    x, a, rate = promote_args_inexact("gamma_cdf", x, shape, rate)
    return gammainc(a, rate * x)


def gamma_logcdf(x, shape, rate) -> Array:
    x, a, rate = promote_args_inexact("gamma_logcdf", x, shape, rate)
    return jnp.log(gamma_cdf(x, a, rate))


def gamma_sf(x, shape, rate) -> Array:
    x, a, rate = promote_args_inexact("gamma_sf", x, shape, rate)
    return 1 - gamma_cdf(x, a, rate)


def gamma_logsf(x, shape, rate) -> Array:
    x, a, rate = promote_args_inexact("gamma_logsf", x, shape, rate)
    return jnp.log(gamma_sf(x, a, rate))


def gamma_ppf(x, shape, rate) -> Array:
    x, a, rate = promote_args_inexact("gamma_ppf", x, shape, rate)
    return igammainv(a, x) / rate


def gamma_isf(x, shape, rate) -> Array:
    x, a, rate = promote_args_inexact("gamma_isf", x, shape, rate)
    return gamma_ppf(1 - x, a, rate)


def gamma_mgf(t, shape, rate) -> Array:
    t, a, rate = promote_args_inexact("gamma_mgf", t, shape, rate)
    return (1 - rate * t) ** (-a)


def gamma_cf(t, shape, rate) -> Array:
    t, a, rate = promote_args_inexact("gamma_cf", t, shape, rate)
    return (1 - rate * 1j * t) ** (-a)
