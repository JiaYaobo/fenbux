import jax.numpy as jnp
import jax.tree_util as jtu

from ._base import identity
from ._func import inverse, log_abs_det_jacobian, transform, value_and_ladj
from ._types import Bijector


class Exp(Bijector):
    """Exp Bijector

    Example:
        >>> import jax.numpy as jnp
        >>> from fenbux.bijectors import Exp, transform
        >>> bij = Exp()
        >>> x = jnp.array([1.0, 2.0, 3.0])
        >>> y = transform(bij, x)
    """

    pass


class Log(Bijector):
    """Log Bijector

    Example:
        >>> import jax.numpy as jnp
        >>> from fenbux.bijectors import Log, transform
        >>> bij = Log()
        >>> x = jnp.array([1.0, 2.0, 3.0])
        >>> y = transform(bij, x)
    """

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


@value_and_ladj.dispatch
def _value_and_ladj(b: Exp, x):
    return transform(b, x), log_abs_det_jacobian(b, x)


@value_and_ladj.dispatch
def _value_and_ladj(b: Log, x):
    return transform(b, x), log_abs_det_jacobian(b, x)
