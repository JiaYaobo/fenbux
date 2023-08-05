from typing import Sequence

from ..tree_utils import tree_add, zeros_like_pytree
from ._base import inverse, log_abs_det_jacobian, transform, value_and_ladj
from ._types import Bijector


class Chain(Bijector):
    """Chain Bijector

    Args:
        bijectors (Sequence[Bijector]): Sequence of bijectors to chain together.

    Example:
        >>> import jax.numpy as jnp
        >>> from fenbux.bijectors import Chain, Exp, Log, Scale, Shift, transform
        >>> bij = Chain([Exp(), Scale(2.0), Log(), Shift(1.0)])
        >>> x = jnp.array([1.0, 2.0, 3.0])
        >>> y = transform(bij, x)
    """

    bijectors: Sequence[Bijector]

    def __init__(self, bijectors: Sequence[Bijector]):
        self.bijectors = bijectors


@transform.dispatch
def _transform(b: Chain, x):
    for bij in reversed(b.bijectors):
        x = transform(bij, x)
    return x


@inverse.dispatch
def _inverse(b: Chain):
    return Chain([inverse(bij) for bij in reversed(b.bijectors)])


@log_abs_det_jacobian.dispatch
def _log_abs_det_jacobian(b: Chain, x):
    ladj = zeros_like_pytree(x)
    for bij in reversed(b.bijectors):
        x, ladj_ = value_and_ladj(bij, x)
        ladj = tree_add(ladj, ladj_)
    return ladj


@value_and_ladj.dispatch
def _value_and_ladj(b: Chain, x):
    return transform(b, x), log_abs_det_jacobian(b, x)
