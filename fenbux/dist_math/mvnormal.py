from jax import numpy as jnp
from jaxtyping import Array


def mvnormal_logpdf(x, mu, cov) -> Array:
    d = jnp.shape(x)[-1]
    x = x - mu

    cov_inv = jnp.linalg.inv(cov)
    return -0.5 * jnp.einsum("...i,...ij,...j->...", x, cov_inv, x) - 0.5 * (
        d * jnp.log(2 * jnp.pi) + jnp.log(jnp.linalg.det(cov))
    )


def mvnormal_pdf(x, mu, cov) -> Array:
    return jnp.exp(mvnormal_logpdf(x, mu, cov))


def mvnormal_mgf(t, mu, cov) -> Array:
    d = jnp.shape(t)[-1]
    t = t - mu
    cov_inv = jnp.linalg.inv(cov)
    return jnp.exp(0.5j * jnp.einsum("...i,...ij,...j->...", t, cov_inv, t)) / jnp.sqrt(
        (2 * jnp.pi) ** d * jnp.linalg.det(cov)
    )


def mvnormal_cf(t, mu, cov) -> Array:
    d = jnp.shape(t)[-1]
    t = t - mu
    cov_inv = jnp.linalg.inv(cov)
    return jnp.exp(0.5j * jnp.einsum("...i,...ij,...j->...", t, cov_inv, t)) / jnp.sqrt(
        (2 * jnp.pi) ** d * jnp.linalg.det(cov)
    )
