import jax.numpy as jnp
from jax.typing import ArrayLike

from ._abstract_impls import evaluate, inverse, is_increasing, ladj, value_and_ladj
from ._inverse import Inverse
from ._typing import Bijector


class Tanh(Bijector):
    """Tanh Bijector

    Example:
        >>> import jax.numpy as jnp
        >>> from fenbux.bijector import Tanh, evaluate
        >>> bij = Tanh()
        >>> x = jnp.array([1.0, 2.0, 3.0])
        >>> y = evaluate(bij, x)
    """

    pass


@evaluate.dispatch
def _evaluate(b: Tanh, x: ArrayLike):
    return jnp.tanh(x)


@inverse.dispatch
def _inverse(b: Tanh):
    return Inverse(b, lambda x: jnp.arctanh(x))


@ladj.dispatch
def _ladj(b: Tanh, x: ArrayLike):
    return jnp.log1p(-jnp.tanh(x) ** 2)


@value_and_ladj.dispatch
def _value_and_ladj(b: Tanh, x: ArrayLike):
    return evaluate(b, x), ladj(b, x)


@is_increasing.dispatch
def _is_increasing(b: Tanh):
    return True
