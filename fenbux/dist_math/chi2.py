from jax import numpy as jnp
from jax.scipy.special import gammainc
from jax.scipy.stats.gamma import logpdf as _jax_gamma_logpdf
from jaxtyping import Array
from tensorflow_probability.substrates.jax.math import igammainv


def chi2_logpdf(x, df) -> Array:
    return _jax_gamma_logpdf(x, df / 2, scale=2.0)


def chi2_pdf(x, df) -> Array:
    return jnp.exp(_jax_gamma_logpdf(x, df / 2, scale=2.0))


def chi2_cdf(x, df) -> Array:
    return gammainc(df / 2, x / 2)


def chi2_logcdf(x, df) -> Array:
    return jnp.log(gammainc(df / 2, x / 2))


def chi2_sf(x, df) -> Array:
    return 1 - gammainc(df / 2, x / 2)


def chi2_logsf(x, df) -> Array:
    return jnp.log(chi2_sf(x, df))


def chi2_ppf(x, df) -> Array:
    return igammainv(df / 2, x) * 2


def chi2_isf(x, df) -> Array:
    return chi2_ppf(1 - x, df)


def chi2_mgf(t, df) -> Array:
    return (1 - 2 * t) ** (-df / 2)


def chi2_cf(t, df) -> Array:
    return (1 - 2 * 1j * t) ** (-df / 2)
