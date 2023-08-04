import jax.numpy as jnp
import jax.tree_util as jtu

from ..core import Shape
from ..tree_utils import tree_reshape
from ._func import inverse, log_abs_det_jacobian, transform, value_and_ladj
from ._types import Bijector


class Reshape(Bijector):
    """Reshape Bijector

    Args:
        in_shape: The input shape.
        out_shape: The output shape.

    Examples:
        >>> import jax.numpy as jnp
        >>> from fenbux.bijectors import Reshape, transform
        >>> bij = Reshape(in_shape=(2, 3), out_shape=(3, 2))
        >>> x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        >>> y = transform(bij, x)
    """
    in_shape: Shape
    out_shape: Shape

    def __init__(self, in_shape: Shape = (-1, ), out_shape: Shape = (-1, )):
        self.in_shape = in_shape
        self.out_shape = out_shape


@inverse.dispatch
def inverse(bij: Reshape):
    return Reshape(bij.out_shape, bij.in_shape)


@transform.dispatch
def transform(bij: Reshape, x):
    return tree_reshape(x, bij.out_shape)


@log_abs_det_jacobian.dispatch
def log_abs_det_jacobian(bij: Reshape, x):
    return jtu.tree_map(lambda _: jnp.zeros_like(x), x)


@value_and_ladj.dispatch
def value_and_ladj(bij: Reshape, x):
    return transform(bij, x), log_abs_det_jacobian(bij, x)
