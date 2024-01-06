from jax import numpy as jnp, pure_callback, ShapeDtypeStruct
from jaxtyping import Array
from scipy.stats import invgauss

from .normal import normal_cdf


def wald_logpdf(x, mu, lam=1) -> Array:
    return jnp.log(
        lam
        / (2 * jnp.pi * x**3) ** 0.5
        * jnp.exp(-lam * (x - mu) ** 2 / (2 * mu**2 * x))
    )


def wald_pdf(x, mu, lam=1) -> Array:
    return jnp.exp(wald_logpdf(x, mu, lam))


def wald_cdf(x, mu, lam=1) -> Array:
    return normal_cdf(jnp.sqrt(lam / x) * (x / mu - 1)) + jnp.exp(
        2 * lam / mu
    ) * normal_cdf(-jnp.sqrt(lam / x) * (x / mu + 1))


def wald_logcdf(x, mu, lam=1) -> Array:
    return jnp.log(wald_cdf(x, mu, lam))


def wald_sf(x, mu, lam=1) -> Array:
    return 1 - wald_cdf(x, mu, lam)


def wald_logsf(x, mu, lam=1) -> Array:
    return jnp.log(wald_sf(x, mu, lam))


def wald_ppf(x, mu):
    def _scipy_callback(x, mu):
        return invgauss(mu).ppf(x)

    result_shape_dtype = ShapeDtypeStruct(shape=jnp.shape(x), dtype=x.dtype)
    return pure_callback(_scipy_callback, result_shape_dtype, x, mu)


def wald_isf(x, mu, lam=1) -> Array:
    return wald_ppf(1 - x, mu, lam)
