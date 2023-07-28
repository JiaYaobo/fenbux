import jax.numpy as jnp
import jax.tree_util as jtu

from ..core import inverse, log_abs_det_jacobian, transform
from ._base import Bijector, identity


class Exp(Bijector):
    pass


class Log(Bijector):
    pass


@inverse.dispatch
def _inverse(b: Exp):
    return Log()


@inverse.dispatch
def _inverse(b: Log):
    return Exp()


@transform.dispatch
def _transform(b: Exp, x):
    return jtu.tree_map(jnp.exp, x)


@transform.dispatch
def _transform(b: Log, x):
    return jtu.tree_map(jnp.log, x)


@log_abs_det_jacobian.dispatch
def _log_abs_det_jacobian(b: Exp, x):
    return jtu.tree_map(identity, x)


@log_abs_det_jacobian.dispatch
def _log_abs_det_jacobian(b: Log, x):
    return jtu.tree_map(lambda x: -jnp.log(x), x)
