import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import PyTree

from .core import KeyArray, PyTreeKey


def split_tree(key: KeyArray, pytree: PyTree, *, is_leaf=None) -> PyTreeKey:
    _, treedef = jtu.tree_flatten(pytree, is_leaf=is_leaf)
    keys = jr.split(key, treedef.num_leaves)
    return jtu.tree_unflatten(treedef, keys)
