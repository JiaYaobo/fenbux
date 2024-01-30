import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import ArrayLike

from ...core import (
    _cdf_impl,
    _cf_impl,
    _intialize_params_tree,
    _kurtosis_impl,
    _logcdf_impl,
    _logpmf_impl,
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
    DTypeLikeInt,
    KeyArray,
    PyTreeVar,
    Shape,
)
from ...dist_math.poisson import (
    poisson_cdf,
    poisson_cf,
    poisson_logcdf,
    poisson_logpmf,
    poisson_mgf,
    poisson_pmf,
    poisson_ppf,
    poisson_sf,
)
from ...random_utils import split_tree
from ...tree_utils import tree_map_dist_at
from .._base import DiscreteUnivariateDistribution


class Poisson(DiscreteUnivariateDistribution):
    """Poisson distribution.

        X ~ Poisson(λ)

    Args:
        rate (PyTree): Rate parameter of the distribution.
        dtype (jax.numpy.dtype): dtype of the distribution, default jnp.float_.
        use_batch (bool): Whether to use with vmap. Default False.

    Examples:
        >>> import jax.numpy as jnp
        >>> from fenbux import logpdf
        >>> from fenbux.univariate import Poisson
        >>> dist = Poisson(1.0)
        >>> logpdf(dist, jnp.ones((10, )))
    """

    rate: PyTreeVar

    def __init__(self, rate=0.0, dtype=jnp.float_, use_batch=False):
        self.rate = _intialize_params_tree(rate, use_batch=use_batch, dtype=dtype)


@_params_impl.dispatch
def _params(d: Poisson):
    return (d.rate,)


@_support_impl.dispatch
def _domain(d: Poisson):
    return jtu.tree_map(lambda r: jnp.zeros_like(r), d.rate), jtu.tree_map(
        lambda r: jnp.full(r, jnp.inf), d.rate
    )


@_mean_impl.dispatch
def _mean(d: Poisson):
    return jtu.tree_map(lambda rate: rate, d.rate)


@_variance_impl.dispatch
def _variance(d: Poisson):
    return jtu.tree_map(lambda rate: rate, d.rate)


@_kurtosis_impl.dispatch
def _kurtosis(d: Poisson):
    return jtu.tree_map(lambda rate: 1 / rate, d.rate)


@_skewness_impl.dispatch
def _skewness(d: Poisson):
    return jtu.tree_map(lambda rate: 1 / jnp.sqrt(rate), d.rate)


@_standard_dev_impl.dispatch
def _standard_dev(d: Poisson):
    return jtu.tree_map(lambda rate: jnp.sqrt(rate), d.rate)


@_logpmf_impl.dispatch
def _logpmf(d: Poisson, x: ArrayLike):
    return tree_map_dist_at(poisson_logpmf, d, x)


@_pmf_impl.dispatch
def _pmf(d: Poisson, x: ArrayLike):
    return tree_map_dist_at(poisson_pmf, d, x)


@_logcdf_impl.dispatch
def _logcdf(d: Poisson, x: ArrayLike):
    return tree_map_dist_at(poisson_logcdf, d, x)


@_cdf_impl.dispatch
def _cdf(d: Poisson, x: ArrayLike):
    return tree_map_dist_at(poisson_cdf, d, x)


@_sf_impl.dispatch
def _sf(d: Poisson, x: ArrayLike):
    return tree_map_dist_at(poisson_sf, d, x)


@_quantile_impl.dispatch
def _quantile(d: Poisson, x: ArrayLike):
    return tree_map_dist_at(poisson_ppf, d, x)

@_mgf_impl.dispatch
def _mgf(d: Poisson, t: ArrayLike):
    return tree_map_dist_at(poisson_mgf, d, t)


@_cf_impl.dispatch
def _cf(d: Poisson, t: ArrayLike):
    return tree_map_dist_at(poisson_cf, d, t)


@_rand_impl.dispatch
def _rand(d: Poisson, key: KeyArray, shape: Shape = (), dtype: DTypeLikeInt = int):
    _key_tree = split_tree(key, d.rate)
    rvs = jtu.tree_map(
        lambda key, r: jr.poisson(key, r, shape=shape, dtype=dtype),
        _key_tree,
        d.rate,
    )
    return rvs

