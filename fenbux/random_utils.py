import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import PyTree


def split_tree(key, pytree: PyTree):
    _, treedef = jtu.tree_flatten(pytree)
    keys = jr.split(key, treedef.num_leaves)
    return jtu.tree_unflatten(treedef, keys)