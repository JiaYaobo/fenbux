from typing import Tuple

from ._abstract_impls import (
    _entropy_impl,
    _kurtosis_impl,
    _mean_impl,
    _params_impl,
    _skewness_impl,
    _standard_dev_impl,
    _support_impl,
    _variance_impl,
)
from ._dist import AbstractDistribution
from ._typing import PyTree


def params(dist: AbstractDistribution) -> Tuple[PyTree, ...]:
    """Extract parameters from a distribution

    Args:
        dist: AbstractDistribution object.

    Example:
        >>> from fenbux import Normal, params
        >>> dist = Normal(0.0, 1.0)
        >>> params(dist)
        (Array(0., dtype=float32), Array(1., dtype=float32))
    """
    return _params_impl(dist)


def support(dist: AbstractDistribution) -> Tuple[PyTree, PyTree]:
    """Support of the distribution

    Args:
        dist: Distribution object.

    Example:
        >>> from fenbux import Normal, support
        >>> dist = Normal(0.0, 1.0)
        >>> support(dist)
        (-inf, inf)
    """
    return _support_impl(dist)


def mean(dist: AbstractDistribution) -> PyTree:
    """Mean of the distribution

    Args:
        dist: Distribution object.

    Example:
        >>> from fenbux import Normal
        >>> dist = Normal(0.0, 1.0)
        >>> mean(dist)
        Array(0., dtype=float32)
    """
    return _mean_impl(dist)


def variance(dist: AbstractDistribution) -> PyTree:
    """Variance of the distribution

    Args:
        dist: Distribution object.

    Example:
        >>> from fenbux import Normal
        >>> dist = Normal(0.0, 1.0)
        >>> variance(dist)
        Array(1., dtype=float32)
    """
    return _variance_impl(dist)


def standard_dev(dist: AbstractDistribution) -> PyTree:
    """Standard deviation of the distribution

    Args:
        dist: Distribution object.

    Example:
        >>> from fenbux import Normal
        >>> dist = Normal(0.0, 1.0)
        >>> standard_dev(dist)
        Array(1., dtype=float32)
    """
    return _standard_dev_impl(dist)


def skewness(dist: AbstractDistribution) -> PyTree:
    """Skewness of the distribution

    Args:
        dist: Distribution object.

    Example:
        >>> from fenbux import Normal, skewness
        >>> dist = Normal(0.0, 1.0)
        >>> skewness(dist)
        Array(0., dtype=float32)
    """
    return _skewness_impl(dist)


def kurtosis(dist: AbstractDistribution) -> PyTree:
    """Kurtosis of the distribution

    Args:
        dist: Distribution object.
    Example:
        >>> from fenbux import Normal, kurtosis
        >>> dist = Normal(0.0, 1.0)
        >>> kurtois(dist)
        Array(0., dtype=float32)
    """
    return _kurtosis_impl(dist)


def entropy(dist: AbstractDistribution) -> PyTree:
    """Entropy of the distribution

    Args:
        dist: Distribution object.

    Example:
        >>> from fenbux import Normal, entropy
        >>> dist = Normal(0.0, 1.0)
        >>> entropy(dist)
        Array(1.4189385, dtype=float32)
    """
    return _entropy_impl(dist)