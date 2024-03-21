import jax.numpy as jnp
from jaxtyping import ArrayLike

from ._abstract_impls import evaluate, inverse, ladj, value_and_ladj
from ._typing import Bijector


class Scale(Bijector):
    """Scale Bijector

    Args:
        scale (ArrayLike): scale parameter

    Example:
        >>> import jax.numpy as jnp
        >>> from fenbux.bijector import Scale, evaluate
        >>> bij = Scale(scale=2.0)
        >>> x = jnp.array([1.0, 2.0, 3.0])
        >>> y = evaluate(bij, x)
    """

    scale: ArrayLike

    def __init__(self, scale: ArrayLike):
        self.scale = scale


@inverse.dispatch
def inverse(bij: Scale):
    return Scale(jnp.invert(bij.scale))


@evaluate.dispatch
def evaluate(bij: Scale, x: ArrayLike):
    return bij.scale * x


@ladj.dispatch
def ladj(bij: Scale, x: ArrayLike):
    return jnp.full_like(x, jnp.log(jnp.abs(bij.scale)))


@value_and_ladj.dispatch
def value_and_ladj(bij: Scale, x: ArrayLike):
    return evaluate(bij, x), ladj(bij, x)
