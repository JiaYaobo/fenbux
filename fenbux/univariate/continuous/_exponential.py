import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

from ...core import (
    _intialize_params_tree,
    cdf,
    cf,
    entropy,
    KeyArray,
    kurtosis,
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
from ...dist_special.exp import (
    exp_cdf,
    exp_cf,
    exp_logcdf,
    exp_logpdf,
    exp_mgf,
    exp_pdf,
    exp_ppf,
    exp_sf,
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
def _params(d: Exponential) -> PyTreeVar:
    return d.rate


@mean.dispatch
def _mean(d: Exponential) -> PyTreeVar:
    return jtu.tree_map(lambda x: 1.0 / x, d.rate)


@variance.dispatch
def _variance(d: Exponential) -> PyTreeVar:
    return jtu.tree_map(lambda x: 1.0 / x**2, d.rate)


@standard_dev.dispatch
def _standard_dev(d: Exponential) -> PyTreeVar:
    return jtu.tree_map(lambda x: 1.0 / x, d.rate)


@skewness.dispatch
def _skewness(d: Exponential) -> PyTreeVar:
    return jtu.tree_map(lambda x: 2.0, d.rate)


@kurtosis.dispatch
def _kurtosis(d: Exponential) -> PyTreeVar:
    return jtu.tree_map(lambda x: 6.0, d.rate)


@support.dispatch
def _support(d: Exponential) -> PyTreeVar:
    return jtu.tree_map(lambda x: (0.0, jnp.inf), d.rate)


@entropy.dispatch
def _entropy(d: Exponential) -> PyTreeVar:
    return jtu.tree_map(lambda x: 1.0 - jnp.log(x), d.rate)


@logpdf.dispatch
def _logpdf(d: Exponential, x: PyTreeVar) -> PyTreeVar:
    return jtu.tree_map(lambda r: _expon_logpdf(x, r), d.rate)


@pdf.dispatch
def _pdf(d: Exponential, x: PyTreeVar) -> PyTreeVar:
    return jtu.tree_map(lambda r: _expon_pdf(x, r), d.rate)


@logcdf.dispatch
def _logcdf(d: Exponential, x: PyTreeVar) -> PyTreeVar:
    return jtu.tree_map(lambda r: _expon_logcdf(x, r), d.rate)


@cdf.dispatch
def _cdf(d: Exponential, x: PyTreeVar) -> PyTreeVar:
    return jtu.tree_map(lambda r: _expon_cdf(x, r), d.rate)


@quantile.dispatch
def _quantile(d: Exponential, x: PyTreeVar) -> PyTreeVar:
    return jtu.tree_map(lambda r: _expon_quantile(x, r), d.rate)


@sf.dispatch
def _sf(d: Exponential, x: PyTreeVar) -> PyTreeVar:
    return jtu.tree_map(lambda r: _expon_sf(x, r), d.rate)


@mgf.dispatch
def _mgf(d: Exponential, t: PyTreeVar) -> PyTreeVar:
    return jtu.tree_map(lambda r: _expon_mgf(t, r), d.rate)


@cf.dispatch
def _cf(d: Exponential, t: PyTreeVar) -> PyTreeVar:
    return jtu.tree_map(lambda r: _expon_cf(t, r), d.rate)


@rand.dispatch
def _rand(
    d: Exponential, key: KeyArray, shape: Shape = (), dtype=jnp.float_
) -> PyTreeVar:
    _key_tree = split_tree(key, d.rate)
    return jtu.tree_map(
        lambda r, k: jr.exponential(k, shape, dtype) / r, d.rate, _key_tree
    )


def _expon_logpdf(x, rate):
    return jtu.tree_map(lambda xx: exp_logpdf(xx, rate), x)


def _expon_pdf(x, rate):
    return jtu.tree_map(lambda xx: exp_pdf(xx, rate), x)


def _expon_logcdf(x, rate):
    return jtu.tree_map(lambda xx: exp_logcdf(xx, rate), x)


def _expon_cdf(x, rate):
    return jtu.tree_map(lambda xx: exp_cdf(xx, rate), x)


def _expon_quantile(x, rate):
    return jtu.tree_map(lambda xx: exp_ppf(xx, rate), x)


def _expon_sf(x, rate):
    return jtu.tree_map(lambda xx: exp_sf(xx, rate), x)


def _expon_mgf(t, rate):
    return jtu.tree_map(lambda tt: exp_mgf(tt, rate), t)


def _expon_cf(t, rate):
    return jtu.tree_map(lambda tt: exp_cf(tt, rate), t)
