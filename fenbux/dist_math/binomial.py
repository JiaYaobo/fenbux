from jax import numpy as jnp, pure_callback, ShapeDtypeStruct
from jax.scipy.special import gammaln, xlog1py, xlogy
from jaxtyping import Array
from scipy.stats import binom

from ..extension import bdtr


def binom_logpmf(x, n, p) -> Array:
    k = jnp.floor(x)
    combiln = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
    return combiln + xlogy(k, p) + xlog1py(n - k, -p)


def binom_pmf(x, n, p) -> Array:
    return jnp.exp(binom_logpmf(x, n, p))


def binom_cdf(x, n, p) -> Array:
    return bdtr(jnp.floor(x), n, p)


def binom_logcdf(x, n, p) -> Array:
    return jnp.log(binom_cdf(x, n, p))


def binom_sf(x, n, p) -> Array:
    return 1 - binom_cdf(x, n, p)


def binom_ppf(x, n, p):
    def _scipy_callback(x, p, n):
        return binom(n, p).ppf(x)

    result_shape_dtype = ShapeDtypeStruct(shape=jnp.shape(x), dtype=x.dtype)
    return pure_callback(_scipy_callback, result_shape_dtype, x, p, n)


def binom_isf(x, n, p):
    return binom_ppf(1 - x, p, n)


def binom_mgf(t, n, p) -> Array:
    return (1 - p + p * jnp.exp(t)) ** n


def binom_cf(t, n, p) -> Array:
    return (1 - p + p * jnp.exp(1j * t)) ** n
