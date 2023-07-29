import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu

from ..core import inverse, log_abs_det_jacobian, transform


def identity(x):
    return x


class Bijector(eqx.Module):
    pass


class Identity(Bijector):
    pass


@inverse.dispatch
def _inverse(b: Identity):
    return Identity()


@transform.dispatch
def _transform(b: Identity, x):
    return x


@log_abs_det_jacobian.dispatch
def _log_abs_det_jacobian(b: Identity, x):
    return jtu.tree_map(lambda xx: jnp.zeros_like(xx), x)
