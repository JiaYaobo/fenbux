import jax.numpy as jnp
import jax.tree_util as jtu

from ..core import inverse, log_abs_det_jacobian, Shape, transform
from ..tree_utils import tree_reshape
from ._base import Bijector


class Reshape(Bijector):
    in_shape: Shape
    out_shape: Shape

    def __init__(self, in_shape: Shape, out_shape: Shape):
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