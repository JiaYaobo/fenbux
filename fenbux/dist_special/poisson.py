from jax import numpy as jnp, pure_callback, ShapeDtypeStruct
from jax._src.numpy.util import promote_args_inexact
from jax.scipy.special import gammainc, gammaln
from jaxtyping import Array
from scipy.stats import poisson


def poisson_logpmf(x, rate) -> Array:
    x, rate = promote_args_inexact("poisson_logpmf", x, rate)
    return x * jnp.log(rate) - gammaln(x + 1) - rate


def poisson_pmf(x, rate) -> Array:
    x, rate = promote_args_inexact("poisson_pmf", x, rate)
    return jnp.exp(poisson_logpmf(x, rate))


def poisson_cdf(x, rate) -> Array:
    x, rate = promote_args_inexact("poisson_cdf", x, rate)
    return 1 - gammainc(jnp.floor(x) + 1, rate)


def poisson_logcdf(x, rate) -> Array:
    x, rate = promote_args_inexact("poisson_logcdf", x, rate)
    return jnp.log(poisson_cdf(x, rate))


def poisson_sf(x, rate) -> Array:
    x, rate = promote_args_inexact("poisson_sf", x, rate)
    return gammainc(jnp.floor(x) + 1, rate)


def poisson_logsf(x, rate) -> Array:
    x, rate = promote_args_inexact("poisson_logsf", x, rate)
    return jnp.log(poisson_sf(x, rate))


def poisson_mgf(t, rate) -> Array:
    t, rate = promote_args_inexact("poisson_mgf", t, rate)
    return jnp.exp(rate * (jnp.exp(t) - 1))


def poisson_cf(t, rate) -> Array:
    t, rate = promote_args_inexact("poisson_cf", t, rate)
    return jnp.exp(rate * (jnp.exp(1j * t) - 1))


def poisson_ppf(x, rate):
    def _scipy_callback(x, rate):
        return poisson(rate).ppf(x)

    x, rate = promote_args_inexact("poisson_ppf", x, rate)
    result_shape_dtype = ShapeDtypeStruct(shape=jnp.shape(x), dtype=x.dtype)
    return pure_callback(_scipy_callback, result_shape_dtype, x, rate)
