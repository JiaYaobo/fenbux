import jax.numpy as jnp
from jaxtyping import ArrayLike

from ..tree_utils import tree_inv
from ._abstract_impls import evaluate, inverse, ladj, value_and_ladj
from ._typing import Bijector


class LeakyReLU(Bijector):
    """LeakyReLU Bijector

    Args:
        alpha (ArrayLike): Slope of the negative part of the function.
            Default value: 0.2.

    Example:
        >>> import jax.numpy as jnp
        >>> from fenbux.bijectors import LeakyReLU, evaluate
        >>> bij = LeakyReLU(0.2)
        >>> x = jnp.array([-1.0, 0.0, 1.0])
        >>> y = evaluate(bij, x)
        Array([-0.2,  0. ,  1. ], dtype=float32)
    """

    alpha: ArrayLike

    def __init__(self, alpha: ArrayLike = 0.2):
        self.alpha = alpha


@inverse.dispatch
def _inverse(b: LeakyReLU) -> LeakyReLU:
    return LeakyReLU(tree_inv(b.alpha))


@evaluate.dispatch
def _evaluate(b: LeakyReLU, x: ArrayLike):
    return jnp.where(x >= 0, x, b.alpha * x)


@ladj.dispatch
def _ladj(b: LeakyReLU, x: ArrayLike):
    return jnp.where(x >= 0, 0.0, jnp.log(jnp.abs(b.alpha)))


@value_and_ladj.dispatch
def _value_and_ladj(b: LeakyReLU, x: ArrayLike):
    return evaluate(b, x), ladj(b, x)
