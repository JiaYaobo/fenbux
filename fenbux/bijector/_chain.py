from typing import Sequence

import jax.numpy as jnp
from jaxtyping import ArrayLike

from ._abstract_impls import evaluate, inverse, is_increasing, ladj, value_and_ladj
from ._typing import Bijector


class Chain(Bijector):
    """Chain Bijector

    Args:
        bijectors (Sequence[Bijector]): Sequence of bijectors to chain together.

    Example:
        >>> import jax.numpy as jnp
        >>> from fenbux.bijector import Chain, Exp, Log, Scale, Shift, evaluate
        >>> bij = Chain([Exp(), Scale(2.0), Log(), Shift(1.0)])
        >>> x = jnp.array([1.0, 2.0, 3.0])
        >>> y = evaluate(bij, x)
    """

    bijectors: Sequence[Bijector]

    def __init__(self, bijectors: Sequence[Bijector]):
        self.bijectors = bijectors


@evaluate.dispatch
def _eval(b: Chain, x: ArrayLike):
    for bij in reversed(b.bijectors):
        x = evaluate(bij, x)
    return x


@inverse.dispatch
def _inverse(b: Chain):
    return Chain([inverse(bij) for bij in reversed(b.bijectors)])


@ladj.dispatch
def _ladj(b: Chain, x: ArrayLike):
    ladj = jnp.zeros_like(x)
    for bij in reversed(b.bijectors):
        x, ladj_ = value_and_ladj(bij, x)
        ladj = ladj + ladj_
    return ladj


@value_and_ladj.dispatch
def _value_and_ladj(b: Chain, x: ArrayLike):
    return evaluate(b, x), ladj(b, x)


@is_increasing.dispatch
def _is_increasing(b: Chain):
    total_non_increasing = sum(not is_increasing(bij) for bij in b.bijectors)
    return total_non_increasing % 2 == 0
