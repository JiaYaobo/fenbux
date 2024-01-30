import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import ArrayLike, PyTree

from ..core import (
    _cdf_impl,
    _cf_impl,
    _check_params_equal_tree_strcutre,
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
    DTypeLikeFloat,
    KeyArray,
    PyTreeVar,
    Shape,
)
from ..dist_math.dirichlet import dirichlet_logpdf, dirichlet_pdf
from ..random_utils import split_tree
from ..tree_utils import _is_multivariate_dist_params, tree_map_dist_at
from ._base import ContinuousMultivariateDistribution


class Dirichlet(ContinuousMultivariateDistribution):
    alpha: PyTreeVar

    """Dirichlet distribution.
        X ~ Dirichlet(α)
        
    Args:
        alpha (PyTreeVar): Shape parameter α.
        dtype (jax.numpy.dtype): dtype of the distribution, default jnp.float_.
        use_batch (bool): Whether to use with vmap. Default False.
        
    Examples:
        >>> import jax.numpy as jnp
        >>> from fenbux import logpdf
        >>> from fenbux.multivariate import Dirichlet
        >>> dist = Dirichlet(jnp.ones((10, )))
        >>> logpdf(dist, jnp.ones((10, )))
    """

    def __init__(self, alpha=0.0, dtype=jnp.float_, use_batch=False):
        _check_params_equal_tree_strcutre(alpha)
        self.alpha = _intialize_params_tree(alpha, use_batch=use_batch, dtype=dtype)


@_params_impl.dispatch
def _params_impl(
    dist: Dirichlet,
):
    return (dist.alpha,)


@_support_impl.dispatch
def _support_impl(
    dist: Dirichlet,
):
    return jtu.tree_map(lambda a: jnp.full_like(a, 0.0), dist.alpha), jtu.tree_map(
        lambda a: jnp.full_like(a, 1.0), dist.alpha
    )


@_mean_impl.dispatch
def _mean_impl(
    dist: Dirichlet,
):
    return jtu.tree_map(
        lambda alpha: alpha / jnp.sum(alpha),
        dist.alpha,
        is_leaf=_is_multivariate_dist_params,
    )


@_logpdf_impl.dispatch
def _dirichlet_logpdf(dist: Dirichlet, x: ArrayLike):
    return tree_map_dist_at(
        dirichlet_logpdf, dist, x, is_leaf_dist=_is_multivariate_dist_params
    )


@_pdf_impl.dispatch
def _dirichlet_pdf(dist: Dirichlet, x: ArrayLike):
    return tree_map_dist_at(
        dirichlet_pdf, dist, x, is_leaf_dist=_is_multivariate_dist_params
    )


@_rand_impl.dispatch
def _rand_dirichlet(
    dist: Dirichlet, key: KeyArray, shape: Shape = (), dtype: DTypeLikeFloat = float
):
    _key_tree = split_tree(key, dist.alpha, is_leaf=_is_multivariate_dist_params)
    return jtu.tree_map(
        lambda a, k: jr.dirichlet(k, a, shape, dtype=dtype),
        dist.alpha,
        _key_tree,
        is_leaf=lambda x: _is_multivariate_dist_params(x) or isinstance(x, KeyArray),
    )
