from typing import Tuple, Union

import equinox as eqx
import jax.numpy as jnp

from ._abstract_impls import _affine_impl, _truncate_impl
from ._dist import AbstractDistribution
from ._typing import DTypeLikeFloat, DTypeLikeInt, KeyArray, PyTree, Shape


@eqx.filter_jit
def affine(d: AbstractDistribution, loc: PyTree = 0.0, scale: PyTree = 1.0) -> PyTree:
    """Affine transformation of a distribution
        y = loc + scale * x
    Args:
        d (AbstractDistribution): A distribution object.
        loc (PyTree): loc parameter of the affine transformation.
        scale (PyTree): scale parameter of the affine transformation.

    Example:
        >>> from fenbux import Normal, affine
        >>> dist = Normal(0.0, 1.0)
        >>> affine(dist, 0.0, 1.0)

    """
    return _affine_impl(d, loc, scale)


@eqx.filter_jit
def truncate(
    d: AbstractDistribution, lower: PyTree = -jnp.inf, upper: PyTree = jnp.inf
) -> PyTree:
    """Truncate a distribution to a given interval.

    Args:
        d (AbstractDistribution): A distribution object.
        lower (PyTree): Lower bound of the truncated distribution.
        upper (PyTree): Upper bound of the truncated distribution.

    Example:
        >>> from fenbux import Normal, truncate
        >>> dist = Normal(0.0, 1.0)
        >>> truncate(dist, -1.0, 1.0)

    """
    raise truncate(d, lower, upper)
