from jax import numpy as jnp, pure_callback, ShapeDtypeStruct
from jax._src.numpy.util import promote_args_inexact
from jaxtyping import Array
from scipy.stats import invgauss

from .normal import normal_cdf


def wald_logpdf(x, mu, lam=1) -> Array:
    x, mu, lam = promote_args_inexact("wald_logpdf", x, mu, lam)
    return jnp.log(
        lam
        / (2 * jnp.pi * x**3) ** 0.5
        * jnp.exp(-lam * (x - mu) ** 2 / (2 * mu**2 * x))
    )


def wald_pdf(x, mu, lam=1) -> Array:
    x, mu, lam = promote_args_inexact("wald_pdf", x, mu, lam)
    return jnp.exp(wald_logpdf(x, mu, lam))


def wald_cdf(x, mu, lam=1) -> Array:
    x, mu, lam = promote_args_inexact("wald_cdf", x, mu, lam)
    return normal_cdf(jnp.sqrt(lam / x) * (x / mu - 1)) + jnp.exp(
        2 * lam / mu
    ) * normal_cdf(-jnp.sqrt(lam / x) * (x / mu + 1))


def wald_logcdf(x, mu, lam=1) -> Array:
    x, mu, lam = promote_args_inexact("wald_logcdf", x, mu, lam)
    return jnp.log(wald_cdf(x, mu, lam))


def wald_sf(x, mu, lam=1) -> Array:
    x, mu, lam = promote_args_inexact("wald_sf", x, mu, lam)
    return 1 - wald_cdf(x, mu, lam)


def wald_logsf(x, mu, lam=1) -> Array:
    x, mu, lam = promote_args_inexact("wald_logsf", x, mu, lam)
    return jnp.log(wald_sf(x, mu, lam))


def wald_ppf(x, mu):
    def _scipy_callback(x, mu):
        return invgauss(mu).ppf(x)

    x, mu = promote_args_inexact("poisson_ppf", x, mu)
    result_shape_dtype = ShapeDtypeStruct(shape=jnp.shape(x), dtype=x.dtype)
    return pure_callback(_scipy_callback, result_shape_dtype, x, mu)


def wald_isf(x, mu, lam=1) -> Array:
    x, mu, lam = promote_args_inexact("wald_isf", x, mu, lam)
    return wald_ppf(1 - x, mu, lam)
