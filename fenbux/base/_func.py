from jaxtyping import PyTree
from plum import Dispatcher


_fenbux_dispatch = Dispatcher()




@_fenbux_dispatch.abstract
def params() -> PyTree:
    """Extract parameters from a distribution
    Example:
    >>> from fenbux import Normal
    >>> n = Normal(0.0, 1.0)
    >>> params(n)
    """
    ...


@_fenbux_dispatch.abstract
def domain():
    """Domain of the distribution
    Example:
    >>> from fenbux import Normal
    >>> n = Normal(0.0, 1.0)
    >>> domain(n)
    """
    ...


@_fenbux_dispatch.abstract
def support():
    """Support of the distribution
    Example:
    >>> from fenbux import Normal
    >>> n = Normal(0.0, 1.0)
    >>> support(n)
    """
    ...


@_fenbux_dispatch.abstract
def entropy():
    """Entropy of the distribution
    Example:
    >>> from fenbux import Normal
    >>> n = Normal(0.0, 1.0)
    >>> entropy(n)
    """
    ...


@_fenbux_dispatch.abstract
def pdf():
    """Probability density function
    Args:
        dist: Distribution object.
        x (ArrayLike): Value to evaluate the pdf at.
    Example:
    >>> from fenbux import Normal
    >>> n = Normal(0.0, 1.0)
    >>> mean(n, 0.0)
    """
    ...


@_fenbux_dispatch.abstract
def logpdf():
    """Log probability density function
    Example:
    >>> from fenbux import Normal
    >>> n = Normal(0.0, 1.0)
    >>> logpdf(n, 0.0)
    """
    ...


@_fenbux_dispatch.abstract
def cdf():
    """Cumulative distribution function
    Example:
    >>> from fenbux import Normal
    >>> n = Normal(0.0, 1.0)
    >>> cdf(n, 0.0)
    """
    ...


@_fenbux_dispatch.abstract
def rand():
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
def quantile():
    """Quantile function
    Example:
    >>> from fenbux import Normal
    >>> n = Normal(0.0, 1.0)
    >>> quantile(n, 0.5)
    """
    ...


@_fenbux_dispatch.abstract
def mean():
    """Expectation of the distribution
    Example:
    >>> from fenbux import Normal
    >>> n = Normal(0.0, 1.0)
    >>> mean(n)
    """
    ...


@_fenbux_dispatch.abstract
def variance():
    """Variance of the distribution
    Example:
    >>> from fenbux import Normal
    >>> n = Normal(0.0, 1.0)
    >>> variance(n)
    """
    ...


@_fenbux_dispatch.abstract
def standard_dev():
    """Standard deviation of the distribution
    Example:
    >>> from fenbux import Normal
    >>> n = Normal(0.0, 1.0)
    >>> standard_dev(n)
    """
    ...


@_fenbux_dispatch.abstract
def skewness():
    """Skewness of the distribution
    Example:
    >>> from fenbux import Normal
    >>> n = Normal(0.0, 1.0)
    >>> skewness(n)
    """
    ...


@_fenbux_dispatch.abstract
def kurtois():
    """Kurtois of the distribution
    Example:
    >>> from fenbux import Normal
    >>> n = Normal(0.0, 1.0)
    >>> kurtois(n)
    """
    ...


@_fenbux_dispatch.abstract
def logpmf():
    """Log probability mass function
    Example:
    >>> from fenbux import Bernoulli
    >>> n = Bernoulli(0.5)
    >>> logpmf(n, 0)
    """
    ...


@_fenbux_dispatch.abstract
def pmf():
    """Probability mass function
    Example:
    >>> from fenbux import Bernoulli
    >>> n = Bernoulli(0.5)
    >>> pmf(n, 0)
    """
    ...


@_fenbux_dispatch.abstract
def mgf():
    """Moment generating function
    Example:
    >>> from fenbux import Normal
    >>> n = Normal(0.0, 1.0)
    >>> mgf(n, 0.5)
    """
    ...


@_fenbux_dispatch.abstract
def sf():
    """Survival function
    Example:
    >>> from fenbux import Normal
    >>> n = Normal(0.0, 1.0)
    >>> sf(n, 0.5)
    """
    ...


@_fenbux_dispatch.abstract
def cf():
    """Characteristic function
    Example:
    >>> from fenbux import Normal
    >>> n = Normal(0.0, 1.0)
    >>> cf(n, 0.5)
    """
    ...


@_fenbux_dispatch.abstract
def inverse():
    ...


@_fenbux_dispatch.abstract
def log_abs_det_jacobian():
    ...
