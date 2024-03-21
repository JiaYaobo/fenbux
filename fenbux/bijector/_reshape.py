import jax.numpy as jnp
from jax.typing import ArrayLike

from ..core import Shape
from ..tree_utils import tree_reshape
from ._abstract_impls import evaluate, inverse, ladj, value_and_ladj
from ._typing import Bijector


class Reshape(Bijector):
    """Reshape Bijector

    Args:
        in_shape: The input shape.
        out_shape: The output shape.

    Examples:
        >>> import jax.numpy as jnp
        >>> from fenbux.bijector import Reshape, evaluate
        >>> bij = Reshape(in_shape=(2, 3), out_shape=(3, 2))
        >>> x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        >>> y = evaluate(bij, x)
    """

    in_shape: Shape
    out_shape: Shape

    def __init__(self, in_shape: Shape = (-1,), out_shape: Shape = (-1,)):
        self.in_shape = in_shape
        self.out_shape = out_shape


@inverse.dispatch
def inverse(bij: Reshape):
    return Reshape(bij.out_shape, bij.in_shape)


@evaluate.dispatch
def evaluate(bij: Reshape, x: ArrayLike):
    return jnp.reshape(x, bij.out_shape)


@ladj.dispatch
def ladj(bij: Reshape, x: ArrayLike):
    return jnp.zeros_like(x)


@value_and_ladj.dispatch
def value_and_ladj(bij: Reshape, x: ArrayLike):
    return evaluate(bij, x), ladj(bij, x)
