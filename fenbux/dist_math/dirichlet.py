from jax import numpy as jnp
from jaxtyping import Array


def dirichlet_logpdf(x, alpha) -> Array:
    return (
        jnp.sum((alpha - 1) * jnp.log(x), axis=-1)
        + jnp.sum(jnp.log(jnp.sum(x, axis=-1)), axis=-1)
        - jnp.sum(jnp.log(alpha), axis=-1)
    )


def dirichlet_pdf(x, alpha) -> Array:
    return jnp.exp(dirichlet_logpdf(x, alpha))
