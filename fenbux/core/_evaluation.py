from jaxtyping import ArrayLike

from ._abstract_impls import (
    _cdf_impl,
    _cf_impl,
    _logcdf_impl,
    _logpdf_impl,
    _logpmf_impl,
    _mgf_impl,
    _pdf_impl,
    _pmf_impl,
    _quantile_impl,
    _rand_impl,
    _sf_impl,
)
from ._dist import AbstractDistribution
from ._typing import DTypeLike, KeyArray, PyTree, Shape


def logpdf(dist: AbstractDistribution, x: ArrayLike) -> PyTree:
    """Log probability density function

    Args:
        dist: Distribution object.
        x (ArrayLike): Value to evaluate the logpdf at.

    Example:
        >>> from fenbux import Normal, logpdf
        >>> dist = Normal(0.0, 1.0)
        >>> logpdf(dist, 0.0)
        Array(-0.9189385, dtype=float32)
    """
    return _logpdf_impl(dist, x)


def logcdf(dist: AbstractDistribution, x: ArrayLike) -> PyTree:
    """Log cumulative distribution function

    Args:
        dist: Distribution object.
        x (ArrayLike): Value to evaluate the logcdf at.

    Example:
        >>> from fenbux import Normal
        >>> dist = Normal(0.0, 1.0)
        >>> logcdf(dist, 0.0)
        Array(-0.6931472, dtype=float32)
    """
    return _logcdf_impl(dist, x)


def logpmf(dist: AbstractDistribution, x: ArrayLike) -> PyTree:
    """Log probability mass function

    Args:
        dist: Distribution object.
        x (ArrayLike): Value to evaluate the logpmf at.

    Example:
        >>> from fenbux import Bernoulli
        >>> dist = Bernoulli(0.5)
        >>> logpmf(dist, 1)
        Array(-0.6931472, dtype=float32)
    """
    return _logpmf_impl(dist, x)


def pdf(dist: AbstractDistribution, x: ArrayLike) -> PyTree:
    """Probability density function

    Args:
        dist: Distribution object.
        x (ArrayLike): Value to evaluate the pdf at.

    Example:
        >>> from fenbux import Normal, pdf
        >>> dist = Normal(0.0, 1.0)
        >>> pdf(dist, 0.0)
        Array(0.3989423, dtype=float32)
    """
    return _pdf_impl(dist, x)


def cdf(dist: AbstractDistribution, x: ArrayLike) -> PyTree:
    """Cumulative distribution function

    Args:
        dist: Distribution object.
        x (ArrayLike): Value to evaluate the cdf at.

    Example:
        >>> from fenbux import Normal, cdf
        >>> dist = Normal(0.0, 1.0)
        >>> cdf(dist, 0.0)
        Array(0.5, dtype=float32)
    """
    return _cdf_impl(dist, x)


def pmf(dist: AbstractDistribution, x: ArrayLike) -> PyTree:
    """Probability mass function

    Args:
        dist: Distribution object.
        x (ArrayLike): Value to evaluate the pmf at.

    Example:
        >>> from fenbux import Bernoulli
        >>> dist = Bernoulli(0.5)
        >>> pmf(dist, 0)
        Array(0.5, dtype=float32)
    """
    return _pmf_impl(dist, x)


def sf(dist: AbstractDistribution, x: ArrayLike) -> PyTree:
    """Survival function of the distribution

    Args:
        dist: Distribution object.
        x (ArrayLike): Value to evaluate the sf at.

    Example:
        >>> from fenbux import Normal, sf
        >>> dist = Normal(0.0, 1.0)
        >>> sf(dist, 0.)
        Array(0.5, dtype=float32)
    """
    return _sf_impl(dist, x)


def quantile(dist: AbstractDistribution, p: ArrayLike) -> PyTree:
    """Quantile function

    Args:
        dist: Distribution object.
        p (ArrayLike): Value to evaluate the quantile at.

    Example:
        >>> from fenbux import Normal
        >>> n = Normal(0.0, 1.0)
        >>> quantile(n, 0.5)
        Array(0., dtype=float32)
    """
    return _quantile_impl(dist, p)


def rand(
    dist: AbstractDistribution,
    key: KeyArray,
    shape: Shape = (),
    dtype: DTypeLike = float,
) -> PyTree:
    """Random number generator

    Args:
        dist: Distribution object.
        key (KeyArray): Random number generator key.
        shape (Shape): Shape of the random number.
        dtype : Data type of the random number.

    Example:
        >>> import jax.random as jr
        >>> from fenbux import Normal
        >>> key = jr.PRNGKey(0)
        >>> dist = Normal(0.0, 1.0)
        >>> rand(dist, key, (2, ))
        Array([-0.20584235,  0.46256348], dtype=float32)
    """
    return _rand_impl(dist, key, shape, dtype)


def cf(dist: AbstractDistribution, t: ArrayLike) -> PyTree:
    """Characteristic function

    Args:
        dist: Distribution object.
        t (ArrayLike): Value to evaluate the cf at.

    Example:
        >>> from fenbux import Normal, cf
        >>> dist = Normal(0.0, 1.0)
        >>> cf(dist, 0.5)
        Array(0.8824969+0.j, dtype=complex64)
    """
    return _cf_impl(dist, t)


def mgf(dist: AbstractDistribution, t: ArrayLike) -> PyTree:
    """Moment generating function

    Args:
        dist: Distribution object.
        t (ArrayLike): Value to evaluate the mgf at.

    Example:
        >>> from fenbux import Normal, mgf
        >>> dist = Normal(0.0, 1.0)
        >>> mgf(dist, 0.5)
        Array(1.1331484, dtype=float32)
    """
    return _mgf_impl(dist, t)
