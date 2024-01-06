from jax import numpy as jnp
from jax.scipy.special import betainc, betaln, xlogy
from jaxtyping import Array

from ..extension import fdtri


def f_logpdf(x, dfn, dfd) -> Array:
    return (
        dfd / 2 * jnp.log(dfd)
        + dfn / 2 * jnp.log(dfn)
        + xlogy(dfn / 2 - 1, x)
        - (((dfn + dfd) / 2) * jnp.log(dfd + dfn * x) + betaln(dfn / 2, dfd / 2))
    )


def f_pdf(x, dfn, dfd) -> Array:
    return jnp.exp(f_logpdf(x, dfn, dfd))


def f_cdf(x, dfn, dfd) -> Array:
    return betainc(dfn / 2, dfd / 2, dfn * x / (dfd + dfn * x))


def f_logcdf(x, dfn, dfd) -> Array:
    return jnp.log(betainc(dfn / 2, dfd / 2, dfn * x / (dfn * x + dfd)))


def f_sf(x, dfn, dfd) -> Array:
    return 1 - betainc(dfn / 2, dfd / 2, dfn * x / (dfd + dfn * x))


def f_logsf(x, dfn, dfd) -> Array:
    return jnp.log(f_sf(x, dfn, dfd))


def f_ppf(x, dfn, dfd) -> Array:
    return fdtri(dfn, dfd, x)


def f_isf(x, dfn, dfd) -> Array:
    return fdtri(dfd, dfn, 1 - x)
