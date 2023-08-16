from typing import Union

from jaxtyping import PyTree
from plum import Dispatcher

from ._dist import AbstractDistribution
from ._types import DTypeLikeFloat, DTypeLikeInt, KeyArray, Shape


_fenbux_dispatch = Dispatcher()


@_fenbux_dispatch.abstract
def _params_impl(dist: AbstractDistribution) -> PyTree:
    """Extract parameters from a distribution

    Args:
        dist: AbstractDistribution object.

    Example:
        >>> from fenbux import Normal, params
        >>> dist = Normal(0.0, 1.0)
        >>> params(dist)
        [Array(0., dtype=float32), Array(1., dtype=float32)]
    """
    ...


@_fenbux_dispatch.abstract
def _support_impl(dist: AbstractDistribution) -> PyTree:
    """Domain of the distribution

    Args:
        dist: Distribution object.

    Example:
        >>> from fenbux import Normal, support
        >>> dist = Normal(0.0, 1.0)
        >>> support(dist)
        (-inf, inf)
    """
    ...


@_fenbux_dispatch.abstract
def _mean_impl(dist: AbstractDistribution) -> PyTree:
    """Mean of the distribution

    Args:
        dist: Distribution object.

    Example:
        >>> from fenbux import Normal
        >>> dist = Normal(0.0, 1.0)
        >>> mean(dist)
        Array(0., dtype=float32)
    """
    ...


@_fenbux_dispatch.abstract
def _variance_impl(dist: AbstractDistribution) -> PyTree:
    """Variance of the distribution

    Args:
        dist: Distribution object.

    Example:
        >>> from fenbux import Normal
        >>> dist = Normal(0.0, 1.0)
        >>> variance(dist)
        Array(1., dtype=float32)
    """
    ...


@_fenbux_dispatch.abstract
def _standard_dev_impl(dist: AbstractDistribution) -> PyTree:
    """Standard deviation of the distribution

    Args:
        dist: Distribution object.

    Example:
        >>> from fenbux import Normal
        >>> dist = Normal(0.0, 1.0)
        >>> standard_dev(dist)
        Array(1., dtype=float32)
    """
    ...


@_fenbux_dispatch.abstract
def _skewness_impl(dist: AbstractDistribution) -> PyTree:
    """Skewness of the distribution

    Args:
        dist: Distribution object.

    Example:
        >>> from fenbux import Normal, skewness
        >>> dist = Normal(0.0, 1.0)
        >>> skewness(dist)
        Array(0., dtype=float32)
    """
    ...


@_fenbux_dispatch.abstract
def _kurtosis_impl(dist: AbstractDistribution) -> PyTree:
    """Kurtosis of the distribution

    Args:
        dist: Distribution object.
    Example:
        >>> from fenbux import Normal, kurtosis
        >>> dist = Normal(0.0, 1.0)
        >>> kurtois(dist)
        Array(0., dtype=float32)
    """
    ...


@_fenbux_dispatch.abstract
def _entropy_impl(dist: AbstractDistribution) -> PyTree:
    """Entropy of the distribution

    Args:
        dist: Distribution object.

    Example:
        >>> from fenbux import Normal, entropy
        >>> dist = Normal(0.0, 1.0)
        >>> entropy(dist)
        Array(1.4189385, dtype=float32)
    """
    ...


@_fenbux_dispatch.abstract
def _pdf_impl(dist: AbstractDistribution, x: PyTree) -> PyTree:
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
    ...


@_fenbux_dispatch.abstract
def _logpdf_impl(dist: AbstractDistribution, x: PyTree) -> PyTree:
    """Log probability density function

    Args:
        dist: Distribution object.
        x (PyTree): Value to evaluate the logpdf at.

    Example:
        >>> from fenbux import Normal, logpdf
        >>> dist = Normal(0.0, 1.0)
        >>> logpdf(dist, 0.0)
        Array(-0.9189385, dtype=float32)
    """
    ...


@_fenbux_dispatch.abstract
def _logcdf_impl(dist: AbstractDistribution, x: PyTree) -> PyTree:
    """Log cumulative distribution function

    Args:
        dist: Distribution object.
        x (PyTree): Value to evaluate the logcdf at.

    Example:
        >>> from fenbux import Normal
        >>> dist = Normal(0.0, 1.0)
        >>> logcdf(dist, 0.0)
        Array(-0.6931472, dtype=float32)
    """
    ...


@_fenbux_dispatch.abstract
def _cdf_impl(dist: AbstractDistribution, x: PyTree) -> PyTree:
    """Cumulative distribution function
    Example:
        >>> from fenbux import Normal
        >>> dist = Normal(0.0, 1.0)
        >>> cdf(dist, 0.0)
        Array(0.5, dtype=float32)
    """
    ...


@_fenbux_dispatch.abstract
def rand(
    dist: AbstractDistribution,
    key: KeyArray,
    shape: Shape,
    dtype: Union[DTypeLikeFloat, DTypeLikeInt],
) -> PyTree:
    """Random number generator

    Args:
        dist: Distribution object.
        key (KeyArray): Random number generator key.
        shape (Shape): Shape of the random number.
        dtype (Union[DTypeLikeFloat, DTypeLikeInt]): Data type of the random number.

    Example:
        >>> import jax.random as jr
        >>> from fenbux import Normal
        >>> key = jr.PRNGKey(0)
        >>> dist = Normal(0.0, 1.0)
        >>> rand(dist, key, (2, ))
        Array([-0.20584235,  0.46256348], dtype=float32)
    """
    ...


@_fenbux_dispatch.abstract
def _quantile_impl(dist: AbstractDistribution, p: PyTree) -> PyTree:
    """Quantile function

    Args:
        dist: Distribution object.
        p (PyTree): Value to evaluate the quantile at.

    Example:
        >>> from fenbux import Normal
        >>> n = Normal(0.0, 1.0)
        >>> quantile(n, 0.5)
        Array(0., dtype=float32)
    """
    ...


@_fenbux_dispatch.abstract
def _logpmf_impl(dist: AbstractDistribution, x: PyTree) -> PyTree:
    """Log probability mass function

    Args:
        dist: Distribution object.
        x (PyTree): Value to evaluate the logpmf at.

    Example:
        >>> from fenbux import Bernoulli
        >>> dist = Bernoulli(0.5)
        >>> logpmf(dist, 1)
        Array(-0.6931472, dtype=float32)
    """
    ...


@_fenbux_dispatch.abstract
def _pmf_impl(dist: AbstractDistribution, x: PyTree) -> PyTree:
    """Probability mass function

    Args:
        dist: Distribution object.
        x (PyTree): Value to evaluate the pmf at.

    Example:
        >>> from fenbux import Bernoulli
        >>> dist = Bernoulli(0.5)
        >>> pmf(dist, 0)
        Array(0.5, dtype=float32)
    """
    ...


@_fenbux_dispatch.abstract
def _mgf_impl(dist: AbstractDistribution, t: PyTree) -> PyTree:
    """Moment generating function

    Args:
        dist: Distribution object.
        t (PyTree): Value to evaluate the mgf at.

    Example:
        >>> from fenbux import Normal, mgf
        >>> dist = Normal(0.0, 1.0)
        >>> mgf(dist, 0.5)
        Array(1.1331484, dtype=float32)
    """
    ...


@_fenbux_dispatch.abstract
def _sf_impl(dist: AbstractDistribution, x: PyTree) -> PyTree:
    """Survival function

    Args:
        dist: Distribution object.
        x (PyTree): Value to evaluate the sf at.

    Example:
        >>> from fenbux import Normal, sf
        >>> dist = Normal(0.0, 1.0)
        >>> sf(dist, 0.)
        Array(0.5, dtype=float32)
    """
    ...


@_fenbux_dispatch.abstract
def _cf_impl(dist: AbstractDistribution, t: PyTree) -> PyTree:
    """Characteristic function

    Args:
        dist: Distribution object.
        t (PyTree): Value to evaluate the cf at.

    Example:
        >>> from fenbux import Normal, cf
        >>> dist = Normal(0.0, 1.0)
        >>> cf(dist, 0.5)
        Array(0.8824969+0.j, dtype=complex64)
    """
    ...


@_fenbux_dispatch.abstract
def affine(d: AbstractDistribution, loc: PyTree, scale: PyTree) -> PyTree:
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
    ...
