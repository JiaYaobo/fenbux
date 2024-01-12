import jax.numpy as jnp
from jaxtyping import ArrayLike

from ._abstract_impls import _affine_impl, _censor_impl, _truncate_impl
from ._dist import AbstractDistribution
from ._typing import PyTree


def affine(
    d: AbstractDistribution, loc: ArrayLike = 0.0, scale: ArrayLike = 1.0
) -> PyTree:
    """Affine transformation of a distribution
        y = loc + scale * x
    Args:
        d (AbstractDistribution): A distribution object.
        loc (ArrayLike): loc parameter of the affine transformation.
        scale (ArrayLike): scale parameter of the affine transformation.

    Example:
        >>> from fenbux import affine, logpdf
        >>> from fenbux.univariate import Normal
        >>> dist = Normal(0.0, 1.0)
        >>> aff_dist = affine(dist, 0.0, 1.0)
        >>> logpdf(aff_dist, 0.0)
    """
    return _affine_impl(d, loc, scale)


def truncate(
    d: AbstractDistribution, lower: ArrayLike = -jnp.inf, upper: ArrayLike = jnp.inf
) -> PyTree:
    """Truncate a distribution to a given interval.

    Args:
        d (AbstractDistribution): A distribution object.
        lower (ArrayLike): Lower bound of the truncated distribution.
        upper (ArrayLike): Upper bound of the truncated distribution.

    Example:
        >>> from fenbux import truncate, logpdf
        >>> from fenbux.univariate import Normal
        >>> dist = Normal(0.0, 1.0)
        >>> truncate(dist, -1.0, 1.0)
        >>> logpdf(dist, -2.0)

    """
    return _truncate_impl(d, lower, upper)


def censor(
    d: AbstractDistribution, lower: ArrayLike = -jnp.inf, upper: ArrayLike = jnp.inf
) -> PyTree:
    """Censor a distribution to a given interval.

    Args:
        d (AbstractDistribution): A distribution object.
        lower (ArrayLike): Lower bound of the censored distribution.
        upper (ArrayLike): Upper bound of the censored distribution.

    Example:
        >>> from fenbux import censor, logpdf
        >>> from fenbux.univariate import Normal
        >>> dist = Normal(0.0, 1.0)
        >>> censor(dist, -1.0, 1.0)
        >>> logpdf(dist, -2.0)

    """
    return _censor_impl(d, lower, upper)
