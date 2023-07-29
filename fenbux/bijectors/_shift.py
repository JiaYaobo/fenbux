import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import ArrayLike

from ..core import inverse, log_abs_det_jacobian, transform
from ..tree_utils import tree_add_array, tree_neg
from ._base import Bijector


class Shift(Bijector):
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
