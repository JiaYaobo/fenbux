from jax import numpy as jnp
from jax.scipy.special import betainc, betaln, xlog1py, xlogy
from jaxtyping import Array


def betabinom_logpmf(x, n, a, b) -> Array:
    k = jnp.floor(x)
    cln = -jnp.log(n + 1) - betaln(n - k + 1, k + 1)
    return cln + betaln(k + a, n - k + b) - betaln(a, b)


def betabinom_pmf(x, n, a, b) -> Array:
    return jnp.exp(betabinom_logpmf(x, n, a, b))
