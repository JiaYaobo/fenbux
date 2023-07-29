import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import ArrayLike

from ..core import inverse, log_abs_det_jacobian, transform
from ..tree_utils import tree_inv, tree_mul_array
from ._base import Bijector


class Scale(Bijector):
    scale: ArrayLike

    def __init__(self, scale: ArrayLike):
        self.scale = scale


@inverse.dispatch
def inverse(bij: Scale):
    return Scale(tree_inv(bij.scale))


@transform.dispatch
def transform(bij: Scale, x):
    return tree_mul_array(bij.scale, x)


@log_abs_det_jacobian.dispatch
def log_abs_det_jacobian(bij: Scale, x):
    return jtu.tree_map(lambda _: jnp.log(jnp.abs(bij.scale)), x)
