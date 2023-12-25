from typing import Callable

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jax.util import safe_map
from jaxtyping import ArrayLike, PyTree

from .core import AbstractDistribution, params, ParamShape, Shape


def broadcast_pytree_arrays_shapes(*args: PyTree) -> PyTree:
    """Broadcast shapes of all leaves of a pytree to a common shape.

    Args:
        *args: PyTree arguments.

    Returns:
        PyTree: PyTree with shapes broadcasted to a common shape.
    """
    if len(args) == 0:
        return args[0]
    tree_list = []
    tree_struct = jtu.tree_structure(args[0])
    for arg in args:
        if jtu.tree_structure(arg) != tree_struct:
            raise ValueError(
                "All arguments must have the same tree structure, got {jtu.tree_structure(arg)} and {shape_tree}"
            )
        tree_list.append(
            jtu.tree_map(lambda x: x if eqx.is_inexact_array_like(x) else None, arg)
        )

    def _broadcast_shape(*args):
        return np.broadcast_shapes(*[np.shape(arg) for arg in args])

    return jtu.tree_map(
        lambda *args: ParamShape(shape=_broadcast_shape(*args)),
        tree_list[0],
        *tree_list[1:],
        is_leaf=eqx.is_inexact_array_like,
    )


def broadcast_pytree_arrays(*args: PyTree) -> PyTree:
    _shapes = broadcast_pytree_arrays_shapes(*args)

    def _broadcast_fn(pytree):
        def _broadcast(x, shape):
            if not eqx.is_inexact_array_like(x):
                return x
            else:
                return jnp.broadcast_to(x, shape)

        return jtu.tree_map(
            lambda l, s: _broadcast(l, s.shape),
            pytree,
            _shapes,
            is_leaf=lambda x: isinstance(x, ParamShape),
        )

    new_args = []
    for arg in args:
        new_args.append(jtu.tree_map(_broadcast_fn, arg))
    return new_args


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


def tree_add(pytree1: PyTree, pytree2: PyTree, is_leaf=None) -> PyTree:
    tree = jtu.tree_map(lambda x, y: x + y, pytree1, pytree2, is_leaf=is_leaf)
    return tree


def tree_add_array(pytree: PyTree, array: np.ndarray, is_leaf=None) -> PyTree:
    tree = jtu.tree_map(lambda x: x + array, pytree, is_leaf=is_leaf)
    return tree


def tree_mul(pytree1: PyTree, pytree2: PyTree, is_leaf=None) -> PyTree:
    tree = jtu.tree_map(lambda x, y: x * y, pytree1, pytree2, is_leaf=is_leaf)
    return tree


def tree_mul_array(pytree: PyTree, array: np.ndarray, is_leaf=None) -> PyTree:
    tree = jtu.tree_map(lambda x: x * array, pytree, is_leaf=is_leaf)
    return tree


def tree_neg(pytree: PyTree, is_leaf=None) -> PyTree:
    tree = jtu.tree_map(lambda x: -x, pytree, is_leaf=is_leaf)
    return tree


def tree_inv(pytree: PyTree, is_leaf=None) -> PyTree:
    tree = jtu.tree_map(lambda x: 1.0 / x, pytree, is_leaf=is_leaf)
    return tree


def tree_reshape(pytree: PyTree, shape: Shape, order="C", is_leaf=None) -> PyTree:
    tree = jtu.tree_map(
        lambda x: jnp.reshape(x, shape, order=order), pytree, is_leaf=is_leaf
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


@eqx.filter_jit
def tree_map_dist_at(
    f: Callable,
    dist: AbstractDistribution,
    x: ArrayLike,
    *,
    is_leaf_dist: Callable = eqx.is_array_like,
) -> PyTree:
    """Apply a function to a distribution at a pytree of points.

    Args:
        f (Callable): Function to be applied to the distribution.
        dist (AbstractDistribution): Distribution to be applied to.
        x (ArrayLike): Points to evaluate the distribution at.
        is_leaf_dist (Callable): If dist parameter treated as a leaf

    Returns:
        PyTree: Result of applying f to dist at x.
    """
    return jtu.tree_map(
        lambda *dist_args: f(x, *dist_args),
        *params(dist),
        is_leaf=is_leaf_dist,
    )


def _is_multivariate_dist_params(p):
    return jnp.ndim(p) >= 1
