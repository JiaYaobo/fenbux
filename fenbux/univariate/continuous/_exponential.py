import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

from ...core import (
    _intialize_params_tree,
    cdf,
    cf,
    entropy,
    KeyArray,
    kurtois,
    logcdf,
    logpdf,
    mean,
    mgf,
    params,
    pdf,
    PyTreeVar,
    quantile,
    rand,
    sf,
    Shape,
    skewness,
    standard_dev,
    support,
    variance,
)
from ...random_utils import split_tree
from .._base import ContinuousUnivariateDistribution


class Exponential(ContinuousUnivariateDistribution):
    """Exponential distribution.

    Args:
        rate (PyTree): Rate parameter.
        dtype (jax.numpy.dtype): dtype of the distribution, default jnp.float_.
        use_batch (bool): Whether to use with vmap. Default False.

    Examples:
        >>> import jax.numpy as jnp
        >>> from fenbux import Exponential, logpdf
        >>> dist = Exponential(1.0)
        >>> logpdf(dist, jnp.ones((10, )))
    """

    rate: PyTreeVar

    def __init__(
        self,
        rate: PyTreeVar = 1.0,
        dtype=jnp.float_,
        use_batch=False,
    ):
        self.rate = _intialize_params_tree(rate, use_batch=use_batch, dtype=dtype)


@params.dispatch
def params(d: Exponential) -> PyTreeVar:
    return d.rate


@mean.dispatch
def mean_(d: Exponential) -> PyTreeVar:
    return jtu.tree_map(lambda x: 1.0 / x, d.rate)


@variance.dispatch
def variance_(d: Exponential) -> PyTreeVar:
    return jtu.tree_map(lambda x: 1.0 / x**2, d.rate)


@standard_dev.dispatch
def standard_dev_(d: Exponential) -> PyTreeVar:
    return jtu.tree_map(lambda x: 1.0 / x, d.rate)


@skewness.dispatch
def skewness_(d: Exponential) -> PyTreeVar:
    return jtu.tree_map(lambda x: 2.0, d.rate)


@kurtois.dispatch
def kurtois_(d: Exponential) -> PyTreeVar:
    return jtu.tree_map(lambda x: 6.0, d.rate)


@support.dispatch
def support_(d: Exponential) -> PyTreeVar:
    return jtu.tree_map(lambda x: (0.0, jnp.inf), d.rate)


@entropy.dispatch
def entropy_(d: Exponential) -> PyTreeVar:
    return jtu.tree_map(lambda x: 1.0 - jnp.log(x), d.rate)


@logpdf.dispatch
def logpdf_(d: Exponential, x: PyTreeVar) -> PyTreeVar:
    return jtu.tree_map(lambda r: _expon_logpdf(x, r), d.rate)


@pdf.dispatch
def pdf_(d: Exponential, x: PyTreeVar) -> PyTreeVar:
    return jtu.tree_map(lambda r: _expon_pdf(x, r), d.rate)


@logcdf.dispatch
def logcdf_(d: Exponential, x: PyTreeVar) -> PyTreeVar:
    return jtu.tree_map(lambda r: _expon_logcdf(x, r), d.rate)


@cdf.dispatch
def cdf_(d: Exponential, x: PyTreeVar) -> PyTreeVar:
    return jtu.tree_map(lambda r: _expon_cdf(x, r), d.rate)


@quantile.dispatch
def quantile_(d: Exponential, x: PyTreeVar) -> PyTreeVar:
    return jtu.tree_map(lambda r: _expon_quantile(x, r), d.rate)


@sf.dispatch
def sf_(d: Exponential, x: PyTreeVar) -> PyTreeVar:
    return jtu.tree_map(lambda r: _expon_sf(x, r), d.rate)


@mgf.dispatch
def mgf_(d: Exponential, t: PyTreeVar) -> PyTreeVar:
    return jtu.tree_map(lambda r: _expon_mgf(t, r), d.rate)


@cf.dispatch
def cf_(d: Exponential, t: PyTreeVar) -> PyTreeVar:
    return jtu.tree_map(lambda r: _expon_cf(t, r), d.rate)


@rand.dispatch
def rand_(
    d: Exponential, key: KeyArray, shape: Shape = (), dtype=jnp.float_
) -> PyTreeVar:
    _key_tree = split_tree(key, d.rate)
    return jtu.tree_map(
        lambda r, k: jr.exponential(k, shape, dtype) / r, d.rate, _key_tree
    )


def _expon_logpdf(x, rate):
    def _fn(x, rate):
        return jnp.log(rate) - rate * x

    return jtu.tree_map(lambda xx: _fn(xx, rate), x)


def _expon_pdf(x, rate):
    def _fn(x, rate):
        return rate * jnp.exp(-rate * x)

    return jtu.tree_map(lambda xx: _fn(xx, rate), x)


def _expon_logcdf(x, rate):
    def _fn(x, rate):
        return jnp.log1p(-jnp.exp(-rate * x))

    return jtu.tree_map(lambda xx: _fn(xx, rate), x)


def _expon_cdf(x, rate):
    def _fn(x, rate):
        return 1.0 - jnp.exp(-rate * x)

    return jtu.tree_map(lambda xx: _fn(xx, rate), x)


def _expon_quantile(x, rate):
    def _fn(x, rate):
        return -jnp.log1p(-x) / rate

    return jtu.tree_map(lambda xx: _fn(xx, rate), x)


def _expon_sf(x, rate):
    def _fn(x, rate):
        return jnp.exp(-rate * x)

    return jtu.tree_map(lambda xx: _fn(xx, rate), x)


def _expon_mgf(t, rate):
    def _fn(t, rate):
        return rate / (rate - t)

    return jtu.tree_map(lambda tt: _fn(tt, rate), t)


def _expon_cf(t, rate):
    def _fn(t, rate):
        return rate / (rate - 1j * t)

    return jtu.tree_map(lambda tt: _fn(tt, rate), t)
