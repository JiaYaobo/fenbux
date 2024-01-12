from jax import numpy as jnp
from jax.scipy.stats import dirichlet
from jaxtyping import Array


def dirichlet_logpdf(x, alpha) -> Array:
    return dirichlet.logpdf(x, alpha)


def dirichlet_pdf(x, alpha) -> Array:
    return dirichlet.pdf(x, alpha)
