from jax import numpy as jnp
from jaxtyping import Array


def weibull_logpdf(x, shape, scale) -> Array:
    return jnp.where(
        x < 0,
        0,
        jnp.log(shape / scale)
        + (shape - 1) * jnp.log(x / scale)
        - (x / scale) ** shape,
    )


def weibull_pdf(x, shape, scale) -> Array:
    return jnp.where(
        x < 0,
        0,
        (shape / scale) * (x / scale) ** (shape - 1) * jnp.exp(-((x / scale) ** shape)),
    )


def weibull_cdf(x, shape, scale) -> Array:
    return jnp.where(
        x < 0,
        0,
        1 - jnp.exp(-((x / scale) ** shape)),
    )


def weibull_logcdf(x, shape, scale) -> Array:
    return jnp.where(
        x < 0,
        0,
        jnp.log1p(-jnp.exp(-((x / scale) ** shape))),
    )


def weibull_sf(x, shape, scale) -> Array:
    return jnp.where(
        x < 0,
        0,
        jnp.exp(-((x / scale) ** shape)),
    )


def weibull_logsf(x, shape, scale) -> Array:
    return jnp.where(
        x < 0,
        0,
        -((x / scale) ** shape),
    )


def weibull_ppf(x, shape, scale) -> Array:
    return scale * (-jnp.log(1 - x)) ** (1 / shape)


def weibull_isf(x, shape, scale) -> Array:
    return weibull_ppf(1 - x, shape, scale)
