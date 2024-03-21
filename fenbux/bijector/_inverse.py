from typing import Callable

from ._abstract_impls import evaluate, inverse
from ._typing import Bijector


class Inverse(Bijector):
    bijector: Bijector
    inv_fn: Callable

    def __init__(self, bijector: Bijector, inv_fn: Callable):
        self.bijector = bijector
        self.inv_fn = inv_fn


@evaluate.dispatch
def _evaluate(b: Inverse, x):
    return b.inv_fn(x)


@inverse.dispatch
def _inverse(b: Inverse):
    return b.b
