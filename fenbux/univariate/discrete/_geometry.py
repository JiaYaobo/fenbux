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
    DTypeLikeFloat,
    KeyArray,
    PyTreeVar,
    Shape,
)
from ...dist_math.geometry import (
    geometric_cdf,
    geometric_logcdf,
    geometric_logpmf,
    geometric_pmf,
    geometric_ppf,
    geometric_sf,
)
from ...random_utils import split_tree
from ...tree_utils import tree_map_dist_at
from .._base import DiscreteUnivariateDistribution


class Geometric(DiscreteUnivariateDistribution):
    p: PyTreeVar

    """Geometric distribution.
    
            X ~ Geometric(p)
            
    Args:
        p (PyTree): Probability of success.
        dtype (jax.numpy.dtype): dtype of the distribution, default jnp.float_.
        use_batch (bool): Whether to use with vmap. Default False.
        
    Examples:
        >>> import jax.numpy as jnp
        >>> from fenbux import logpmf
        >>> from fenbux.univariate import Geometric
        >>> dist = Geometric(0.5)
        >>> logpmf(dist, jnp.ones((10, )))
    """

    def __init__(self, p=0.0, dtype=jnp.float_, use_batch=False):
        self.p = _intialize_params_tree(p, use_batch=use_batch, dtype=dtype)


@_params_impl.dispatch
def _params_impl(dist: Geometric):
    return (dist.p,)


@_support_impl.dispatch
def _support_impl(dist: Geometric):
    return jtu.tree_map(lambda p: jnp.zeros_like(p), dist.p), jtu.tree_map(
        lambda p: jnp.full_like(p, jnp.inf), dist.p
    )


@_mean_impl.dispatch
def _mean_impl(dist: Geometric):
    return jtu.tree_map(lambda p: 1.0 / p, dist.p)


@_variance_impl.dispatch
def _variance_impl(dist: Geometric):
    return jtu.tree_map(lambda p: (1.0 - p) / (p**2), dist.p)


@_standard_dev_impl.dispatch
def _standard_dev_impl(dist: Geometric):
    return jtu.tree_map(lambda p: jnp.sqrt((1.0 - p) / (p**2)), dist.p)


@_skewness_impl.dispatch
def _skewness_impl(dist: Geometric):
    return jtu.tree_map(lambda p: (2.0 - p) / jnp.sqrt(1.0 - p), dist.p)


@_kurtosis_impl.dispatch
def _kurtosis_impl(dist: Geometric):
    return jtu.tree_map(lambda p: 6.0 + (p**2) / (1.0 - p), dist.p)


@_pmf_impl.dispatch
def _pmf_impl(dist: Geometric, x: ArrayLike):
    return tree_map_dist_at(geometric_pmf, dist, x)


@_logpmf_impl.dispatch
def _logpmf_impl(dist: Geometric, x: ArrayLike):
    return tree_map_dist_at(geometric_logpmf, dist, x)


@_cdf_impl.dispatch
def _cdf_impl(dist: Geometric, x: ArrayLike):
    return tree_map_dist_at(geometric_cdf, dist, x)


@_logcdf_impl.dispatch
def _logcdf_impl(dist: Geometric, x: ArrayLike):
    return tree_map_dist_at(geometric_logcdf, dist, x)


@_sf_impl.dispatch
def _sf_impl(dist: Geometric, x: ArrayLike):
    return tree_map_dist_at(geometric_sf, dist, x)


@_quantile_impl.dispatch
def _quantile_impl(dist: Geometric, x: ArrayLike):
    return tree_map_dist_at(geometric_ppf, dist, x)


@_rand_impl.dispatch
def _rand(d: Geometric, key: KeyArray, shape: Shape = (), dtype: DTypeLikeFloat = int):
    _key_tree = split_tree(key, d.n)
    rvs = jtu.tree_map(
        lambda p, k: jr.geometric(k, p, shape=shape, dtype=dtype),
        d.p,
        _key_tree,
    )
    return rvs
