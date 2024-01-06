from jax import numpy as jnp
from jax.scipy.special import betainc, gammaln
from jaxtyping import Array
from tensorflow_probability.substrates.jax.math import betaincinv


def t_logpdf(x, df) -> Array:
    return (
        gammaln((df + 1) / 2)
        - gammaln(df / 2)
        - 0.5 * jnp.log(df * jnp.pi)
        - 0.5 * (df + 1) * jnp.log1p(x**2 / df)
    )


def t_pdf(x, df) -> Array:
    return jnp.exp(t_logpdf(x, df))


def t_cdf(x, df) -> Array:
    return 0.5 * (
        1.0 + jnp.sign(x) - jnp.sign(x) * betainc(df / 2, 0.5, df / (df + x**2))
    )


def t_logcdf(x, df) -> Array:
    return jnp.log(t_cdf(x, df))


def t_sf(x, df) -> Array:
    return 1 - 0.5 * (
        1.0 + jnp.sign(x) - jnp.sign(x) * betainc(df / 2, 0.5, df / (df + x**2))
    )


def t_ppf(x, df) -> Array:
    beta_val = betaincinv(df / 2, 0.5, 1 - jnp.abs(2 * x - 1))
    return jnp.sqrt(df * (1 - beta_val) / beta_val) * jnp.sign(x - 0.5)


