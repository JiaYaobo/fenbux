from typing import Callable

from jaxtyping import ArrayLike

from ._abstract_impls import evaluate, inverse, is_increasing, ladj
from ._typing import Bijector


class Inverse(Bijector):
    """ Inverse Bijector
    
    Args:
        bijector (Bijector): Bijector to invert.
        inv_fn (Callable): Inverse function.
        
    Example:
        >>> import jax.numpy as jnp
        >>> from fenbux.bijector import Inverse, Exp, evaluate
        >>> bij = Inverse(Exp(), jnp.log)
        >>> x = jnp.array([1.0, 2.0, 3.0])
        >>> y = evaluate(bij, x)
    
    """
    bijector: Bijector
    inv_fn: Callable

    def __init__(self, bijector: Bijector, inv_fn: Callable):
        self.bijector = bijector
        self.inv_fn = inv_fn


@evaluate.dispatch
def _evaluate(b: Inverse, x: ArrayLike):
    return b.inv_fn(x)


@inverse.dispatch
def _inverse(b: Inverse):
    return b.b


@is_increasing.dispatch
def _is_increasing(b: Inverse):
    return is_increasing(b.bijector)


@ladj.dispatch
def _ladj(b: Inverse, x: ArrayLike):
    return -ladj(b.bijector, b.inv_fn(x))
