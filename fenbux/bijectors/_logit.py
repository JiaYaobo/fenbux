import jax.numpy as jnp
import jax.tree_util as jtu
from jax.scipy.special import expit, logit
from jax.typing import ArrayLike

from ._base import Inverse
from ._func import inverse, log_abs_det_jacobian, transform, value_and_ladj
from ._types import Bijector


class Logit(Bijector):
    a: ArrayLike
    b: ArrayLike

    def __init__(self, a: ArrayLike = 0.0, b: ArrayLike = 1.0):
        self.a = a
        self.b = b


def _logit(x, a, b):
    return logit((x - a) / (b - a))


def _ilogit(x, a, b):
    return a + (b - a) * expit(x)


@transform.dispatch
def _transform(b: Logit, x):
    return jtu.tree_map(lambda xx: _logit(xx, b.a, b.b), x)


@inverse.dispatch
def _inverse(b: Logit):
    return Inverse(b, lambda x: jtu.tree_map(lambda xx: _ilogit(xx, b.a, b.b), x))


@log_abs_det_jacobian.dispatch
def _log_abs_det_jacobian(b: Logit, x):
    return jtu.tree_map(lambda xx: -jnp.log((xx - b.a) * (b.b - xx) / (b.b - b.a)), x)


@value_and_ladj.dispatch
def _value_and_ladj(b: Logit, x):
    return transform(b, x), log_abs_det_jacobian(b, x)
