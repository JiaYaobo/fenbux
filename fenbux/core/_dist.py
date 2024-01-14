import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jax.dtypes import canonicalize_dtype

from ._typing import PyTreeVar, Shape


class ParamShape(eqx.Module):
    shape: Shape


class AbstractDistribution(eqx.Module):
    def broadcast_shapes(self):
        """Broadcast shape for each of distribution parameters.
        Example:
        >>> from fenbux import Normal
        >>> n = Normal(np.ones(2, ), np.ones(10, 2))
        >>> n.broadcast_shapes()
        ((10, 2), (10, 2))
        """
        tree_list = []
        for _, item in self.__dict__.items():
            if isinstance(item, PyTreeVar):
                tree_list.append(
                    jtu.tree_map(
                        lambda x: x if eqx.is_inexact_array_like(x) else None, item
                    )
                )

        def _broadcast_shape(*args):
            return np.broadcast_shapes(*[np.shape(arg) for arg in args])

        return jtu.tree_map(
            lambda *args: ParamShape(shape=_broadcast_shape(*args)),
            tree_list[0],
            *tree_list[1:],
            is_leaf=eqx.is_inexact_array_like,
        )

    def broadcast_to(self, shape: Shape, is_leaf=eqx.is_inexact_array_like):
        """Broadcast all distribution parameters to a common shape on each leaf level.
        Example:
        >>> from fenbux import Normal
        >>> n = Normal(np.ones(2, ), np.ones(10, 2))
        >>> n.broadcast_to((10, 2))
        """

        attr = jtu.tree_map(
            lambda leave: jnp.broadcast_to(leave, shape), self, is_leaf=is_leaf
        )
        return attr

    def broadcast_params(self, is_leaf=eqx.is_inexact_array_like):
        """Broadcast all distribution parameters to a common shape on each leaf level.
        Example:
        >>> from fenbux import Normal
        >>> dist = Normal(np.ones(2, ), np.ones(10, 2))
        >>> dist.broadcast_params()
        """
        _shapes = self.broadcast_shapes()
        _tree = self

        def _broadcast_fn(pytree):
            def _broadcast(x, shape):
                if not is_leaf(x):
                    return x
                else:
                    return jnp.broadcast_to(x, shape)

            return jtu.tree_map(
                lambda l, s: _broadcast(l, s.shape),
                pytree,
                _shapes,
                is_leaf=lambda x: isinstance(x, ParamShape),
            )

        for key, item in self.__dict__.items():
            if isinstance(item, PyTreeVar):
                _tree = eqx.tree_at(
                    lambda x: x.__getattribute__(key), _tree, replace_fn=_broadcast_fn
                )
        return _tree

    def broadcast_params_leaves(self):
        return jtu.tree_leaves(self.broadcast_params())

    def broadcast_params_structure(self):
        return jtu.tree_structure(self.broadcast_params())

    @property
    def broadcast_shape_(self):
        return jtu.tree_leaves(self.broadcast_shapes())


def _is_none(x):
    return x is None


def _check_params_equal_tree_strcutre(*args, use_batch=False):
    if use_batch:
        return
    _strct = jtu.tree_structure(args[0])
    for arg in args[1:]:
        if jtu.tree_structure(arg, is_leaf=_is_none) != _strct:
            raise ValueError(
                f"all input params must have the same tree structure, got {jtu.tree_structure(args[0], is_leaf=_is_none)} and {_strct}"
            )


def _intialize_params_tree(*args, use_batch=False, dtype=float):
    new_args = []
    if use_batch:
        for arg in args:
            new_args.append(jtu.tree_map(lambda x: int(x), arg))
        return new_args
    else:
        dtype = canonicalize_dtype(dtype)
        for arg in args:
            new_args.append(jtu.tree_map(lambda x: jnp.asarray(x, dtype=dtype), arg))
        return new_args[0] if len(new_args) == 1 else new_args
