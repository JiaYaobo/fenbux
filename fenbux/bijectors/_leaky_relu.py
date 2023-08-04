import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import ArrayLike

from ..tree_utils import tree_inv
from ._func import inverse, log_abs_det_jacobian, transform, value_and_ladj
from ._types import Bijector


class LeakyReLU(Bijector):
    alpha: ArrayLike

    def __init__(self, alpha: ArrayLike = 0.2):
        self.alpha = alpha


@inverse.dispatch
def _inverse(b: LeakyReLU) -> LeakyReLU:
    return LeakyReLU(tree_inv(b.alpha))


@transform.dispatch
def _transform(b: LeakyReLU, x):
    return jtu.tree_map(lambda xx: jnp.where(xx >= 0, xx, b.alpha * xx), x)


@log_abs_det_jacobian.dispatch
def _log_abs_det_jacobian(b: LeakyReLU, x):
    return jtu.tree_map(
        lambda xx: jnp.where(xx >= 0, 0.0, jnp.log(jnp.abs(b.alpha))), x
    )


@value_and_ladj.dispatch
def _value_and_ladj(b: LeakyReLU, x):
    return transform(b, x), log_abs_det_jacobian(b, x)
