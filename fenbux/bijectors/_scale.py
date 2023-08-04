import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import ArrayLike

from ..tree_utils import tree_inv, tree_mul_array
from ._func import inverse, log_abs_det_jacobian, transform, value_and_ladj
from ._types import Bijector


class Scale(Bijector):
    """Scale Bijector

    Args:
        scale (ArrayLike): scale parameter

    Example:
        >>> import jax.numpy as jnp
        >>> from fenbux.bijectors import Scale, transform
        >>> bij = Scale(scale=2.0)
        >>> x = jnp.array([1.0, 2.0, 3.0])
        >>> y = transform(bij, x)
    """
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


@value_and_ladj.dispatch
def value_and_ladj(bij: Scale, x):
    return transform(bij, x), log_abs_det_jacobian(bij, x)
