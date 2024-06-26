import jax.numpy as jnp
from jax.scipy.special import expit, logit
from jax.typing import ArrayLike

from ._abstract_impls import evaluate, inverse, is_increasing, ladj, value_and_ladj
from ._inverse import Inverse
from ._typing import Bijector


class Logit(Bijector):
    """Logit Bijector

    Args:
        a (ArrayLike): Lower bound of the input domain.
            Default value: 0.0.
        b (ArrayLike): Upper bound of the input domain.
            Default value: 1.0.
            
    Example:
        >>> import jax.numpy as jnp
        >>> from fenbux.bijector import Logit, evaluate
        >>> bij = Logit()
        >>> x = jnp.array([0.1, 0.5, 0.9])
        >>> y = evaluate(bij, x)
    """
    a: ArrayLike
    b: ArrayLike

    def __init__(self, a: ArrayLike = 0.0, b: ArrayLike = 1.0):
        self.a = a
        self.b = b


def _logit(x, a, b):
    return logit((x - a) / (b - a))


def _ilogit(x, a, b):
    return a + (b - a) * expit(x)


@evaluate.dispatch
def _evaluate(b: Logit, x: ArrayLike):
    return _logit(x, b.a, b.b)


@inverse.dispatch
def _inverse(b: Logit):
    return Inverse(b, lambda x: _ilogit(x, b.a, b.b))


@ladj.dispatch
def _ladj(b: Logit, x: ArrayLike):
    return -jnp.log((x - b.a) * (b.b - x) / (b.b - b.a))


@value_and_ladj.dispatch
def _value_and_ladj(b: Logit, x: ArrayLike):
    return evaluate(b, x), ladj(b, x)


@is_increasing.dispatch
def _is_increasing(b: Logit):
    return True
