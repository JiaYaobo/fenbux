import jax.numpy as jnp
from jaxtyping import ArrayLike

from ._abstract_impls import evaluate, inverse, is_increasing, ladj, value_and_ladj
from ._typing import Bijector


def identity(x):
    return x


class Identity(Bijector):
    pass


@inverse.dispatch
def _inverse(b: Identity):
    return Identity()


@evaluate.dispatch
def _evaluate(b: Identity, x: ArrayLike):
    return x


@ladj.dispatch
def _ladj(b: Identity, x: ArrayLike):
    return jnp.zeros_like(x)


@value_and_ladj.dispatch
def _value_and_ladj(b: Identity, x: ArrayLike):
    return evaluate(b, x), ladj(b, x)


@is_increasing.dispatch
def _is_increasing(b: Identity):
    return True
