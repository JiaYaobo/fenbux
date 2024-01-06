from jax import numpy as jnp, pure_callback, ShapeDtypeStruct
from jax.scipy.special import gammainc, gammaln
from jaxtyping import Array
from scipy.stats import poisson


def poisson_logpmf(x, rate) -> Array:
    return x * jnp.log(rate) - gammaln(x + 1) - rate


def poisson_pmf(x, rate) -> Array:
    return jnp.exp(poisson_logpmf(x, rate))


def poisson_cdf(x, rate) -> Array:
    return 1 - gammainc(jnp.floor(x) + 1, rate)


def poisson_logcdf(x, rate) -> Array:
    return jnp.log(poisson_cdf(x, rate))


def poisson_sf(x, rate) -> Array:
    return gammainc(jnp.floor(x) + 1, rate)


def poisson_logsf(x, rate) -> Array:
    return jnp.log(poisson_sf(x, rate))


def poisson_mgf(t, rate) -> Array:
    return jnp.exp(rate * (jnp.exp(t) - 1))


def poisson_cf(t, rate) -> Array:
    return jnp.exp(rate * (jnp.exp(1j * t) - 1))


def poisson_ppf(x, rate):
    def _scipy_callback(x, rate):
        return poisson(rate).ppf(x)

    result_shape_dtype = ShapeDtypeStruct(shape=jnp.shape(x), dtype=x.dtype)
    return pure_callback(_scipy_callback, result_shape_dtype, x, rate)
