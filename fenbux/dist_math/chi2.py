from jax import numpy as jnp
from jax._src.numpy.util import promote_args_inexact
from jax.scipy.special import gammainc
from jax.scipy.stats.gamma import logpdf as _jax_gamma_logpdf
from jaxtyping import Array
from tensorflow_probability.substrates.jax.math import igammainv


def chi2_logpdf(x, df) -> Array:
    x, df = promote_args_inexact("chi2_logpdf", x, df)
    return _jax_gamma_logpdf(x, df / 2, scale=2.0)


def chi2_pdf(x, df) -> Array:
    x, df = promote_args_inexact("chi2_pdf", x, df)
    return jnp.exp(_jax_gamma_logpdf(x, df / 2, scale=2.0))


def chi2_cdf(x, df) -> Array:
    x, df = promote_args_inexact("chi2_cdf", x, df)
    return gammainc(df / 2, x / 2)


def chi2_logcdf(x, df) -> Array:
    x, df = promote_args_inexact("chi2_logcdf", x, df)
    return jnp.log(gammainc(df / 2, x / 2))


def chi2_sf(x, df) -> Array:
    x, df = promote_args_inexact("chi2_sf", x, df)
    return 1 - gammainc(df / 2, x / 2)


def chi2_logsf(x, df) -> Array:
    x, df = promote_args_inexact("chi2_logsf", x, df)
    return jnp.log(chi2_sf(x, df))


def chi2_ppf(x, df) -> Array:
    x, df = promote_args_inexact("chi2_ppf", x, df)
    return igammainv(df / 2, x) * 2


def chi2_isf(x, df) -> Array:
    x, df = promote_args_inexact("chi2_isf", x, df)
    return chi2_ppf(1 - x, df)


def chi2_mgf(t, df) -> Array:
    t, df = promote_args_inexact("chi2_mgf", t, df)
    return (1 - 2 * t) ** (-df / 2)


def chi2_cf(t, df) -> Array:
    t, df = promote_args_inexact("chi2_cf", t, df)
    return (1 - 2 * 1j * t) ** (-df / 2)
