from jaxtyping import PyTree
from plum import Dispatcher

from ._dist import AbstractDistribution


_fenbux_dispatch = Dispatcher()


@_fenbux_dispatch.abstract
def params(dist: AbstractDistribution) -> PyTree:
    """Extract parameters from a distribution

    Args:
        dist: AbstractDistribution object.

    Example:
    >>> from fenbux import Normal, params
    >>> n = Normal(0.0, 1.0)
    >>> params(n)
    [Array(0., dtype=float32), Array(1., dtype=float32)]
    """
    ...


@_fenbux_dispatch.abstract
def support(dist: AbstractDistribution) -> PyTree:
    """Domain of the distribution

    Args:
        dist: Distribution object.

    Example:
    >>> from fenbux import Normal, support
    >>> n = Normal(0.0, 1.0)
    >>> support(n)
    """
    ...


@_fenbux_dispatch.abstract
def entropy(dist: AbstractDistribution) -> PyTree:
    """Entropy of the distribution
    Example:
    >>> from fenbux import Normal, entropy
    >>> n = Normal(0.0, 1.0)
    >>> entropy(n)
    """
    ...


@_fenbux_dispatch.abstract
def pdf(dist: AbstractDistribution, x: PyTree) -> PyTree:
    """Probability density function
    Args:
        dist: Distribution object.
        x (ArrayLike): Value to evaluate the pdf at.
    Example:
    >>> from fenbux import Normal, pdf
    >>> n = Normal(0.0, 1.0)
    >>> pdf(n, 0.0)
    """
    ...


@_fenbux_dispatch.abstract
def logpdf(dist: AbstractDistribution, x: PyTree) -> PyTree:
    """Log probability density function
    Example:
    >>> from fenbux import Normal, logpdf
    >>> n = Normal(0.0, 1.0)
    >>> logpdf(n, 0.0)
    """
    ...


@_fenbux_dispatch.abstract
def logcdf(dist: AbstractDistribution, x: PyTree) -> PyTree:
    """Log cumulative distribution function
    Example:
    >>> from fenbux import Normal
    >>> n = Normal(0.0, 1.0)
    >>> logcdf(n, 0.0)
    """
    ...


@_fenbux_dispatch.abstract
def cdf(dist: AbstractDistribution, x: PyTree) -> PyTree:
    """Cumulative distribution function
    Example:
    >>> from fenbux import Normal
    >>> n = Normal(0.0, 1.0)
    >>> cdf(n, 0.0)
    """
    ...


@_fenbux_dispatch.abstract
def rand(dist: AbstractDistribution, key: PyTree, shape) -> PyTree:
    """Random number generator
    Example:
    >>> import jax.random as jr
    >>> from fenbux import Normal
    >>> key = jr.PRNGKey(0)
    >>> n = Normal(0.0, 1.0)
    >>> rand(n, key, (10, 10))
    """
    ...


@_fenbux_dispatch.abstract
def quantile(dist: AbstractDistribution, q: PyTree) -> PyTree:
    """Quantile function
    Example:
    >>> from fenbux import Normal
    >>> n = Normal(0.0, 1.0)
    >>> quantile(n, 0.5)
    Array(0., dtype=float32)
    """
    ...


@_fenbux_dispatch.abstract
def mean(dist: AbstractDistribution) -> PyTree:
    """Expectation of the distribution
    Example:
    >>> from fenbux import Normal
    >>> n = Normal(0.0, 1.0)
    >>> mean(n)
    Array(0., dtype=float32)
    """
    ...


@_fenbux_dispatch.abstract
def variance(dist: AbstractDistribution) -> PyTree:
    """Variance of the distribution
    Example:
    >>> from fenbux import Normal
    >>> n = Normal(0.0, 1.0)
    >>> variance(n)
    Array(1., dtype=float32)
    """
    ...


@_fenbux_dispatch.abstract
def standard_dev(dist: AbstractDistribution) -> PyTree:
    """Standard deviation of the distribution
    Example:
    >>> from fenbux import Normal
    >>> n = Normal(0.0, 1.0)
    >>> standard_dev(n)
    Array(1., dtype=float32)
    """
    ...


@_fenbux_dispatch.abstract
def skewness(dist: AbstractDistribution) -> PyTree:
    """Skewness of the distribution
    Example:
    >>> from fenbux import Normal, skewness
    >>> n = Normal(0.0, 1.0)
    >>> skewness(n)
    Array(0., dtype=float32)
    """
    ...


@_fenbux_dispatch.abstract
def kurtois(dist: AbstractDistribution) -> PyTree:
    """Kurtois of the distribution
    Example:
    >>> from fenbux import Normal, kurtois
    >>> n = Normal(0.0, 1.0)
    >>> kurtois(n)
    Array(0., dtype=float32)
    """
    ...


@_fenbux_dispatch.abstract
def logpmf(dist: AbstractDistribution, x: PyTree) -> PyTree:
    """Log probability mass function
    Example:
    >>> from fenbux import Bernoulli
    >>> n = Bernoulli(0.5)
    >>> logpmf(n, 0)
    """
    ...


@_fenbux_dispatch.abstract
def pmf(dist: AbstractDistribution, x: PyTree) -> PyTree:
    """Probability mass function
    Example:
    >>> from fenbux import Bernoulli
    >>> n = Bernoulli(0.5)
    >>> pmf(n, 0)
    """
    ...


@_fenbux_dispatch.abstract
def mgf(dist: AbstractDistribution, t: PyTree) -> PyTree:
    """Moment generating function

    Args:
        dist: Distribution object.
        t (PyTree): Value to evaluate the mgf at.

    Example:
    >>> from fenbux import Normal, mgf
    >>> n = Normal(0.0, 1.0)
    >>> mgf(n, 0.5)
    Array(1.1331484, dtype=float32)
    """
    ...


@_fenbux_dispatch.abstract
def sf(dist: AbstractDistribution, x: PyTree) -> PyTree:
    """Survival function
    Example:
    >>> from fenbux import Normal, sf
    >>> n = Normal(0.0, 1.0)
    >>> sf(n, 0.5)
    """
    ...


@_fenbux_dispatch.abstract
def cf(dist: AbstractDistribution, t: PyTree) -> PyTree:
    """Characteristic function

    Args:
        dist: Distribution object.
        t (PyTree): Value to evaluate the cf at.

    Example:
    >>> from fenbux import Normal, cf
    >>> n = Normal(0.0, 1.0)
    >>> cf(n, 0.5)
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
    """
    ...


@_fenbux_dispatch.abstract
def inverse():
    ...


@_fenbux_dispatch.abstract
def log_abs_det_jacobian():
    ...


@_fenbux_dispatch.abstract
def transform():
    ...
