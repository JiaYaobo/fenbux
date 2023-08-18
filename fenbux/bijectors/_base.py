from typing import Callable

import jax.numpy as jnp
import jax.tree_util as jtu

from ._abstract_impls import inverse, log_abs_det_jacobian, transform, value_and_ladj
from ._typing import Bijector


def identity(x):
    return x


class Identity(Bijector):
    pass


class Inverse(Bijector):
    d: Bijector
    inv_fn: Callable

    def __init__(self, d: Bijector):
        self.d = d
        self.inv_fn = Callable

    def __call__(self, x):
        return self.inv_fn(x)


@inverse.dispatch
def _inverse(b: Identity):
    return Identity()


@transform.dispatch
def _transform(b: Identity, x):
    return x


@log_abs_det_jacobian.dispatch
def _log_abs_det_jacobian(b: Identity, x):
    return jtu.tree_map(lambda xx: jnp.zeros_like(xx), x)


@value_and_ladj.dispatch
def _value_and_ladj(b: Identity, x):
    return transform(b, x), log_abs_det_jacobian(b, x)
