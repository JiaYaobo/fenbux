import jax.numpy as jnp
from jaxtyping import ArrayLike

from ._abstract_impls import evaluate, inverse, ladj, value_and_ladj
from ._typing import Bijector


class Shift(Bijector):
    """Shift Bijector

    Args:
        shift (ArrayLike): shift parameter

    Example:
        >>> import jax.numpy as jnp
        >>> from fenbux.bijector import Shift, evaluate
        >>> bij = Shift(shift=2.0)
        >>> x = jnp.array([1.0, 2.0, 3.0])
        >>> y = evaluate(bij, x)
    """

    shift: ArrayLike

    def __init__(self, shift: ArrayLike):
        self.shift = shift


@inverse.dispatch
def inverse(bij: Shift):
    return Shift(-bij.shift)


@evaluate.dispatch
def evaluate(bij: Shift, x):
    return bij.shift + x


@ladj.dispatch
def ladj(bij: Shift, x):
    return jnp.zeros_like(x)


@value_and_ladj.dispatch
def value_and_ladj(bij: Shift, x):
    return evaluate(bij, x), ladj(bij, x)
