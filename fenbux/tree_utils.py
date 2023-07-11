import jax.numpy as jnp
import jax.tree_util as jtu
from jax.core import safe_map
from jaxtyping import PyTree

from .base import ParamShape


def broadcast_pytree_to(pytree: PyTree, shapeTree: PyTree) -> PyTree:
    attr = jtu.tree_map(
        lambda l, s: jnp.broadcast_to(l, s.shape),
        pytree,
        shapeTree,
        is_leaf=lambda x: isinstance(x, ParamShape),
    )
    return attr


def zeros_pytree(shapeTree: PyTree) -> PyTree:
    tree = jtu.tree_map(
        lambda s: jnp.zeros(s.shape),
        shapeTree,
        is_leaf=lambda x: isinstance(x, ParamShape),
    )
    return tree


def ones_pytree(shapeTree: PyTree) -> PyTree:
    tree = jtu.tree_map(
        lambda s: jnp.ones(s.shape),
        shapeTree,
        is_leaf=lambda x: isinstance(x, ParamShape),
    )
    return tree


def full_pytree(shapeTree: PyTree, value):
    tree = jtu.tree_map(
        lambda s: jnp.full(s.shape, value),
        shapeTree,
        is_leaf=lambda x: isinstance(x, ParamShape),
    )
    return tree


def zeros_like_pytree(pytree: PyTree, is_leaf=None) -> PyTree:
    tree = jtu.tree_map(lambda leave: jnp.zeros_like(leave), pytree, is_leaf=is_leaf)
    return tree


def ones_like_pytree(pytree: PyTree, is_leaf=None) -> PyTree:
    tree = jtu.tree_map(lambda leave: jnp.ones_like(leave), pytree, is_leaf=is_leaf)
    return tree


def full_like_pytree(pytree: PyTree, value, is_leaf=None) -> PyTree:
    tree = jtu.tree_map(
        lambda leave: jnp.full_like(leave, value), pytree, is_leaf=is_leaf
    )
    return tree


def tree_structures_all_eq(*args, is_leaf=None, **kwargs) -> bool:
    for arg in args:
        if jtu.tree_structure(arg, is_leaf=is_leaf) != jtu.tree_structure(
            args[0], is_leaf=is_leaf
        ):
            return False
    if kwargs:
        kwargs_first_key = list(kwargs.keys())[0]
        for _, arg in kwargs.items():
            if args:
                if jtu.tree_structure(arg, is_leaf=is_leaf) != jtu.tree_structure(
                    args[0], is_leaf=is_leaf
                ):
                    return False
            else:
                if jtu.tree_structure(arg, is_leaf=is_leaf) != jtu.tree_structure(
                    kwargs[kwargs_first_key], is_leaf=is_leaf
                ):
                    return False

    return True


def tree_map(fn, *args, is_leaf=None, flat_kwargnames=None, **kwargs) -> PyTree:
    """Converts a function that takes a flat list of arguments into one that takes a pytree of arguments.

    Args:
        fn (Callable): Function that takes a flat list of arguments.
        *args: Arguments to be passed to fn.
        is_leaf (Callable): Function that determines whether a node is a leaf or not.
        flat_kwargnames (List[str]): List of keyword argument names to be passed to fn which are not pytrees.
        **kwargs: Keyword arguments to be passed to fn.

    Returns:
        PyTree: Result of fn applied to the pytree of arguments.
    """

    flat_dict = {}
    if flat_kwargnames is not None:
        flat_args = [kwargs.pop(argname) for argname in flat_kwargnames]
        flat_dict = dict(zip(flat_kwargnames, flat_args))

    if not tree_structures_all_eq(*args, **kwargs, is_leaf=is_leaf):
        raise ValueError("args and kwargs must have the same tree structure")

    kwargs_keys = kwargs.keys()
    no_tree_inputs = False
    if args or kwargs:
        tree_struct = (
            jtu.tree_structure(args[0])
            if args
            else jtu.tree_structure(kwargs[list(kwargs_keys)[0]])
        )
    else:
        no_tree_inputs = True

    args = safe_map(jtu.tree_leaves, args)
    kwargs_vals = safe_map(lambda k: jtu.tree_leaves(kwargs[k]), kwargs_keys)
    flatten_out = []

    kws = []
    for _kwargs_vals in zip(*kwargs_vals):
        kws.append(dict(zip(kwargs_keys, _kwargs_vals)))

    if len(args) == 0 and len(kwargs) != 0:
        for kw in kws:
            flatten_out.append(fn(**kw, **flat_dict))
    elif len(kwargs) == 0 and len(args) != 0:
        for _arg in zip(*args):
            flatten_out.append(fn(*_arg, **flat_dict))
    elif len(kwargs) == 0 and len(args) == 0:
        flatten_out.append(fn(**flat_dict))
    else:
        for *_args, _kw in zip(*args, kws):
            flatten_out.append(fn(*_args, **_kw, **flat_dict))

    if no_tree_inputs:
        return flatten_out[0]
    return jtu.tree_unflatten(tree_struct, flatten_out)
