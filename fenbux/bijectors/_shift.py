import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import ArrayLike

from ..tree_utils import tree_add_array, tree_neg
from ._func import inverse, log_abs_det_jacobian, transform, value_and_ladj
from ._types import Bijector


class Shift(Bijector):
    """Shift Bijector

    Args:
        shift (ArrayLike): shift parameter

    Example:
        >>> import jax.numpy as jnp
        >>> from fenbux.bijectors import Shift, transform
        >>> bij = Shift(shift=2.0)
        >>> x = jnp.array([1.0, 2.0, 3.0])
        >>> y = transform(bij, x)
    """

    shift: ArrayLike

    def __init__(self, shift: ArrayLike):
        self.shift = shift


@inverse.dispatch
def inverse(bij: Shift):
    return Shift(tree_neg(bij.shift))


@transform.dispatch
def transform(bij: Shift, x):
    return tree_add_array(bij.shift, x)


@log_abs_det_jacobian.dispatch
def log_abs_det_jacobian(bij: Shift, x):
    return jtu.tree_map(lambda x: jnp.zeros_like(x), x)


@value_and_ladj.dispatch
def value_and_ladj(bij: Shift, x):
    return transform(bij, x), log_abs_det_jacobian(bij, x)
