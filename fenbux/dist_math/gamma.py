from jax import numpy as jnp
from jax.scipy.special import gammainc
from jax.scipy.stats.gamma import logpdf as _jax_gamma_logpdf
from jaxtyping import Array
from tensorflow_probability.substrates.jax.math import igammainv


def gamma_logpdf(x, shape, rate) -> Array:
    return _jax_gamma_logpdf(x, shape, scale=1 / rate)


def gamma_pdf(x, shape, rate) -> Array:
    return jnp.exp(gamma_logpdf(x, shape, rate))


def gamma_cdf(x, shape, rate) -> Array:
    return gammainc(shape, rate * x)


def gamma_logcdf(x, shape, rate) -> Array:
    return jnp.log(gamma_cdf(x, shape, rate))


def gamma_sf(x, shape, rate) -> Array:
    return 1 - gamma_cdf(x, shape, rate)


def gamma_logsf(x, shape, rate) -> Array:
    return jnp.log(gamma_sf(x, shape, rate))


def gamma_ppf(x, shape, rate) -> Array:
    return igammainv(shape, x) / rate


def gamma_isf(x, shape, rate) -> Array:
    return gamma_ppf(1 - x, shape, rate)


def gamma_mgf(t, shape, rate) -> Array:
    return (1 - rate * t) ** (-shape)


def gamma_cf(t, shape, rate) -> Array:
    return (1 - rate * 1j * t) ** (-shape)
