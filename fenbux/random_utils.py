from typing import Callable

import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import PyTree

from .core import KeyArray, PyTreeKey


def split_tree(key: KeyArray, pytree: PyTree, *, is_leaf: Callable = None) -> PyTreeKey:
    """Split a key into a pytree of keys.

    Args:
        key (KeyArray): A jax.random.KeyArray.
        pytree (PyTree): A pytree to split.
        is_leaf (Callable): A function to determine whether a pytree node is a leaf. Default is jtu.is_leaf.

    Returns:
        PyTreeKey: A pytree of keys.
    """
    _, treedef = jtu.tree_flatten(pytree, is_leaf=is_leaf)
    keys = jr.split(key, treedef.num_leaves)
    return jtu.tree_unflatten(treedef, keys)
