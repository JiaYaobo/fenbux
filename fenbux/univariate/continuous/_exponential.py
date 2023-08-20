import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

from ...core import (
    _cdf_impl,
    _cf_impl,
    _entropy_impl,
    _intialize_params_tree,
    _kurtosis_impl,
    _logcdf_impl,
    _logpdf_impl,
    _mean_impl,
    _mgf_impl,
    _params_impl,
    _pdf_impl,
    _quantile_impl,
    _rand_impl,
    _sf_impl,
    _skewness_impl,
    _standard_dev_impl,
    _support_impl,
    _variance_impl,
    KeyArray,
    PyTreeVar,
    Shape,
)
from ...dist_math.exp import (
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


@_params_impl.dispatch
def _params(d: Exponential) -> PyTreeVar:
    return d.rate


@_support_impl.dispatch
def _support(d: Exponential) -> PyTreeVar:
    return jtu.tree_map(lambda r: jnp.zeros_like(r), d.rate), jtu.tree_map(
        lambda r: jnp.full_like(r, jnp.inf), d.rate
    )


@_mean_impl.dispatch
def _mean(d: Exponential) -> PyTreeVar:
    return jtu.tree_map(lambda x: 1.0 / x, d.rate)


@_variance_impl.dispatch
def _variance(d: Exponential) -> PyTreeVar:
    return jtu.tree_map(lambda x: 1.0 / x**2, d.rate)


@_standard_dev_impl.dispatch
def _standard_dev(d: Exponential) -> PyTreeVar:
    return jtu.tree_map(lambda x: 1.0 / x, d.rate)


@_skewness_impl.dispatch
def _skewness(d: Exponential) -> PyTreeVar:
    return jtu.tree_map(lambda x: 2.0, d.rate)


@_kurtosis_impl.dispatch
def _kurtosis(d: Exponential) -> PyTreeVar:
    return jtu.tree_map(lambda x: 6.0, d.rate)


@_support_impl.dispatch
def _support(d: Exponential) -> PyTreeVar:
    return jtu.tree_map(lambda x: (0.0, jnp.inf), d.rate)


@_entropy_impl.dispatch
def _entropy(d: Exponential) -> PyTreeVar:
    return jtu.tree_map(lambda x: 1.0 - jnp.log(x), d.rate)


@_logpdf_impl.dispatch
def _logpdf(d: Exponential, x: PyTreeVar) -> PyTreeVar:
    return jtu.tree_map(lambda r: _expon_logpdf(x, r), d.rate)


@_pdf_impl.dispatch
def _pdf(d: Exponential, x: PyTreeVar) -> PyTreeVar:
    return jtu.tree_map(lambda r: _expon_pdf(x, r), d.rate)


@_logcdf_impl.dispatch
def _logcdf(d: Exponential, x: PyTreeVar) -> PyTreeVar:
    return jtu.tree_map(lambda r: _expon_logcdf(x, r), d.rate)


@_cdf_impl.dispatch
def _cdf(d: Exponential, x: PyTreeVar) -> PyTreeVar:
    return jtu.tree_map(lambda r: _expon_cdf(x, r), d.rate)


@_quantile_impl.dispatch
def _quantile(d: Exponential, x: PyTreeVar) -> PyTreeVar:
    return jtu.tree_map(lambda r: _expon_quantile(x, r), d.rate)


@_sf_impl.dispatch
def _sf(d: Exponential, x: PyTreeVar) -> PyTreeVar:
    return jtu.tree_map(lambda r: _expon_sf(x, r), d.rate)


@_mgf_impl.dispatch
def _mgf(d: Exponential, t: PyTreeVar) -> PyTreeVar:
    return jtu.tree_map(lambda r: _expon_mgf(t, r), d.rate)


@_cf_impl.dispatch
def _cf(d: Exponential, t: PyTreeVar) -> PyTreeVar:
    return jtu.tree_map(lambda r: _expon_cf(t, r), d.rate)


@_rand_impl.dispatch
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
