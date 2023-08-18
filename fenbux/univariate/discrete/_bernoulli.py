import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jax import lax

from ...core import (
    _cdf_impl,
    _cf_impl,
    _entropy_impl,
    _intialize_params_tree,
    _kurtosis_impl,
    _logcdf_impl,
    _mean_impl,
    _mgf_impl,
    _params_impl,
    _pmf_impl,
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
from ...dist_math.bernoulli import (
    bernoulli_cdf,
    bernoulli_cf,
    bernoulli_logcdf,
    bernoulli_logpmf,
    bernoulli_mgf,
    bernoulli_pmf,
    bernoulli_ppf,
    bernoulli_sf,
)
from ...random_utils import split_tree
from .._base import DiscreteUnivariateDistribution


class Bernoulli(DiscreteUnivariateDistribution):
    """Bernoulli distribution.
        X ~ Bernoulli(p)
    Args:
        p (PyTree): Probability of success.
        dtype (jax.numpy.dtype): dtype of the distribution, default jnp.float_.
        use_batch (bool): Whether to use with vmap. Default False.

    Examples:
        >>> import jax.numpy as jnp
        >>> from fenbux import Bernoulli, logpdf
        >>> dist = Bernoulli(0.5)
        >>> logpdf(dist, jnp.ones((10, )))
    """

    p: PyTreeVar

    def __init__(self, p=0.0, dtype=jnp.float_, use_batch=False):
        self.p = _intialize_params_tree(p, use_batch=use_batch, dtype=dtype)


@_params_impl.dispatch
def _params(d: Bernoulli):
    return d.p


@_support_impl.dispatch
def _support(d: Bernoulli):
    return jtu.tree_map(lambda _: {0, 1}, d.p)


@_mean_impl.dispatch
def _mean(d: Bernoulli):
    return jtu.tree_map(lambda p: p, d.p)


@_variance_impl.dispatch
def _variance(d: Bernoulli):
    return jtu.tree_map(lambda p: p * (1 - p), d.p)


@_standard_dev_impl.dispatch
def _standard_dev(d: Bernoulli):
    return jtu.tree_map(lambda p: jnp.sqrt(p * (1 - p)), d.p)


@_kurtosis_impl.dispatch
def _kurtosis(d: Bernoulli):
    return jtu.tree_map(
        lambda p: jnp.where(
            lax.gt(p, 0.0) & lax.lt(p, 1.0),
            (1 - 6 * p * (1 - p)) / (p * (1 - p)),
            jnp.nan,
        ),
        d.p,
    )


@_skewness_impl.dispatch
def _skewness(d: Bernoulli):
    return jtu.tree_map(
        lambda p: jnp.where(
            lax.gt(p, 0.0) & lax.lt(p, 1.0),
            (1 - 2 * p) / jnp.sqrt(p * (1 - p)),
            jnp.nan,
        ),
        d.p,
    )


@_entropy_impl.dispatch
def _entropy(d: Bernoulli):
    return jtu.tree_map(lambda p: -p * jnp.log(p) - (1 - p) * jnp.log(1 - p), d.p)


@_pmf_impl.dispatch
def _pmf(d: Bernoulli, x: PyTreeVar):
    return jtu.tree_map(lambda p: _bernoulli_pmf(p, x), d.p)


@_rand_impl.dispatch
def _rand(d: Bernoulli, key: KeyArray, shape: Shape = (), dtype=jnp.float_):
    _key_tree = split_tree(key, d.p)
    rvs = jtu.tree_map(
        lambda p, k: jr.bernoulli(k, p, shape=shape, dtype=dtype),
        d.p,
        _key_tree,
    )
    return rvs


@_cdf_impl.dispatch
def _cdf(d: Bernoulli, x: PyTreeVar):
    return jtu.tree_map(lambda p: _bernoulli_cdf(p, x), d.p)


@_logcdf_impl.dispatch
def _logcdf(d: Bernoulli, x: PyTreeVar):
    return jtu.tree_map(lambda p: _bernoulli_log_cdf(p, x), d.p)


@_quantile_impl.dispatch
def _quantile(d: Bernoulli, x: PyTreeVar):
    return jtu.tree_map(lambda p: _bernoulli_quantile(p, x), d.p)


@_mgf_impl.dispatch
def _mgf(d: Bernoulli, t: PyTreeVar):
    return jtu.tree_map(lambda p: _bernoulli_mgf(p, t), d.p)


@_cf_impl.dispatch
def _cf(d: Bernoulli, t: PyTreeVar):
    return jtu.tree_map(lambda p: _bernoulli_cf(p, t), d.p)


@_sf_impl.dispatch
def _sf(d: Bernoulli, x: PyTreeVar):
    return jtu.tree_map(lambda p: _bernoulli_sf(p, x), d.p)


def _bernoulli_pmf(p, x):
    return jtu.tree_map(lambda xx: bernoulli_pmf(xx, p), x)


def _bernoulli_cdf(p, x):
    return jtu.tree_map(lambda xx: bernoulli_cdf(xx, p), x)


def _bernoulli_quantile(p, x):
    return jtu.tree_map(lambda xx: bernoulli_ppf(xx, p), x)


def _bernoulli_mgf(p, t):
    return jtu.tree_map(lambda tt: bernoulli_mgf(tt, p), t)


def _bernoulli_cf(p, t):
    return jtu.tree_map(lambda tt: bernoulli_cf(tt, p), t)


def _bernoulli_sf(p, x):
    return jtu.tree_map(lambda xx: bernoulli_sf(xx, p), x)


def _bernoulli_log_cdf(p, x):
    return jtu.tree_map(lambda xx: bernoulli_logcdf(xx, p), x)
