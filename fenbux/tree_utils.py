import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import PyTree

from .base import ParamShape


def broadcast_pytree_to(pytree, shape):
    attr = jtu.tree_map(
        lambda l, s: jnp.broadcast_to(l, s.shape),
        pytree,
        shape,
        is_leaf=lambda x: isinstance(x, ParamShape),
    )
    return attr


def zeros_pytree(shape):
    tree = jtu.tree_map(
        lambda s: jnp.zeros(s.shape), shape, is_leaf=lambda x: isinstance(x, ParamShape)
    )
    return tree


def ones_pytree(shape):
    tree = jtu.tree_map(
        lambda s: jnp.ones(s.shape), shape, is_leaf=lambda x: isinstance(x, ParamShape)
    )
    return tree


def full_pytree(shape: PyTree, value):
    tree = jtu.tree_map(
        lambda s: jnp.full(s.shape, value), shape, is_leaf=lambda x: isinstance(x, ParamShape)
    )
    return tree


def zeros_like_pytree(pytree: PyTree):
    tree = jtu.tree_map(lambda leave: jnp.zeros_like(leave), pytree)
    return tree


def ones_like_pytree(pytree: PyTree):
    tree = jtu.tree_map(lambda leave: jnp.ones_like(leave), pytree)
    return tree


def full_like_pytree(pytree: PyTree, value):
    tree = jtu.tree_map(lambda leave: jnp.full_like(leave, value), pytree)
    return tree
