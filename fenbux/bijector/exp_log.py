import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import ArrayLike, Float, PyTree

from ..base import inverse, log_abs_det_jacobian
from .bijector import Bijector


class Exp(Bijector):
    def __call__(self, x: PyTree[Float[ArrayLike, "..."]], /):
        return jtu.tree_map(jnp.exp, x)


class Log(Bijector):
    def __call__(self, x: PyTree[Float[ArrayLike, "..."]], /):
        return jtu.tree_map(jnp.log, x)


@inverse.dispatch
def _inverse(b: Exp, x: PyTree[Float[ArrayLike, "..."]]):
    return jtu.tree_map(jnp.log, x)


@inverse.dispatch
def _inverse(b: Exp):
    return Log()


@inverse.dispatch
def _inverse(b: Log, x: PyTree[Float[ArrayLike, "..."]]):
    return jtu.tree_map(jnp.exp, x)


@inverse.dispatch
def _inverse(b: Log):
    return Exp()


@log_abs_det_jacobian.dispatch
def _log_abs_det_jacobian(b: Exp, x: PyTree[Float[ArrayLike, "..."]]):
    return jtu.tree_map(jnp.log, x)


@log_abs_det_jacobian.dispatch
def _log_abs_det_jacobian(b: Log, x: PyTree[Float[ArrayLike, "..."]]):
    return jtu.tree_map(jnp.exp, x)
