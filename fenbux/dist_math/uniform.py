from jax import numpy as jnp
from jaxtyping import Array


def uniform_logpdf(x, lower, upper) -> Array:
    return jnp.where(
        (x < lower) | (x > upper),
        -jnp.inf,
        -jnp.log(upper - lower),
    )


def uniform_pdf(x, lower, upper) -> Array:
    return jnp.where(
        (x < lower) | (x > upper),
        0,
        1 / (upper - lower),
    )


def uniform_cdf(x, lower, upper) -> Array:
    return jnp.where(
        x < lower,
        0,
        jnp.where(
            x > upper,
            1,
            (x - lower) / (upper - lower),
        ),
    )


def uniform_logcdf(x, lower, upper) -> Array:
    return jnp.where(
        x < lower,
        -jnp.inf,
        jnp.where(
            x > upper,
            0,
            jnp.log((x - lower) / (upper - lower)),
        ),
    )


def uniform_sf(x, lower, upper) -> Array:
    return jnp.where(
        x < lower,
        1,
        jnp.where(
            x > upper,
            0,
            (upper - x) / (upper - lower),
        ),
    )


def uniform_logsf(x, lower, upper) -> Array:
    return jnp.where(
        x < lower,
        0,
        jnp.where(
            x > upper,
            -jnp.inf,
            jnp.log((upper - x) / (upper - lower)),
        ),
    )


def uniform_ppf(x, lower, upper) -> Array:
    return jnp.where(
        (x < 0) | (x > 1),
        jnp.nan,
        lower + x * (upper - lower),
    )


def uniform_isf(x, lower, upper) -> Array:
    return jnp.where(
        (x < 0) | (x > 1),
        jnp.nan,
        upper - x * (upper - lower),
    )


def uniform_mgf(t, lower, upper) -> Array:
    return jnp.where(
        t == 0,
        1,
        (jnp.exp(t * upper) - jnp.exp(t * lower)) / (t * (upper - lower)),
    )


def uniform_cf(t, lower, upper) -> Array:
    return jnp.where(
        t == 0,
        1,
        (jnp.exp(1j * t * upper) - jnp.exp(1j * t * lower))
        / (1j * t * (upper - lower)),
    )
