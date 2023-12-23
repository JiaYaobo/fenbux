from typing import Tuple, Union

import equinox as eqx

from ._abstract_impls import _affine_impl
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
