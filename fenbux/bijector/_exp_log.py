import jax.numpy as jnp
from jaxtyping import ArrayLike

from ._abstract_impls import (
    evaluate,
    inverse,
    ladj,
    value_and_ladj,
)
from ._identity import identity
from ._typing import Bijector


class Exp(Bijector):
    """Exp Bijector

    Example:
        >>> import jax.numpy as jnp
        >>> from fenbux.bijectors import Exp, evaluate
        >>> bij = Exp()
        >>> x = jnp.array([1.0, 2.0, 3.0])
        >>> y = evaluate(bij, x)
    """

    pass


class Log(Bijector):
    """Log Bijector

    Example:
        >>> import jax.numpy as jnp
        >>> from fenbux.bijectors import Log, evaluate
        >>> bij = Log()
        >>> x = jnp.array([1.0, 2.0, 3.0])
        >>> y = evaluate(bij, x)
    """

    pass


@inverse.dispatch
def _inverse(b: Exp):
    return Log()


@inverse.dispatch
def _inverse(b: Log):
    return Exp()


@evaluate.dispatch
def _evaluate(b: Exp, x: ArrayLike):
    return jnp.exp(x)


@evaluate.dispatch
def _evaluate(b: Log, x: ArrayLike):
    return jnp.log(x)


@ladj.dispatch
def _ladj(b: Exp, x: ArrayLike):
    return identity(x)


@ladj.dispatch
def _ladj(b: Log, x: ArrayLike):
    return -jnp.log(x)



@value_and_ladj.dispatch
def _value_and_ladj(b: Exp, x: ArrayLike):
    return evaluate(b, x), ladj(b, x)
