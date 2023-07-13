import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from ._types import PyTreeVar, Shape


class DistributionParam(eqx.Module):
    val: PyTreeVar


class ParamShape(eqx.Module):
    shape: Shape


class AbstractDistribution(eqx.Module):
    def broadcast_shapes(self):
        """Broadcast shape for all counterparts of each DistributionParam.
        Example:
        >>> from fenbux import Normal
        >>> n = Normal(np.ones(2, ), np.ones(10, 2))
        >>> n.broadcast_shapes()
        ((10, 2), (10, 2))
        """
        tree_list = []
        for _, item in self.__dict__.items():
            tree_list.append(
                jtu.tree_map(
                    lambda x: x if eqx.is_inexact_array_like(x) else None, item
                )
            )

        return jtu.tree_map(
            lambda *args: ParamShape(
                np.broadcast_shapes(*[np.shape(arg) for arg in args])
            ),
            tree_list[0],
            *tree_list[1:],
            is_leaf= eqx.is_inexact_array_like
        )

    def broadcast_to(self, shape: Shape, is_leaf=eqx.is_inexact_array_like):
        """Broadcast all distribution parameters to a common shape on each leaf level.
        Example:
        >>> from fenbux import Normal
        >>> n = Normal(np.ones(2, ), np.ones(10, 2))
        >>> n.broadcast_to((10, 2))
        """

        attr = jtu.tree_map(lambda leave: jnp.broadcast_to(leave, shape), self, is_leaf=is_leaf)
        return attr

    def broadcast_params(self):
        """Broadcast all distribution parameters to a common shape on each leaf level.
        Example:
        >>> from fenbux import Normal
        >>> n = Normal(np.ones(2, ), np.ones(10, 2))
        >>> n.broadcast_params()
        """
        _shapes = self.broadcast_shapes()
        _tree = self

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
    

class AbstractDistributionTransform(eqx.Module): ...
