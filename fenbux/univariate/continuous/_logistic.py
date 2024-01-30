import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import ArrayLike

from ...core import (
    _cdf_impl,
    _check_params_equal_tree_strcutre,
    _entropy_impl,
    _intialize_params_tree,
    _kurtosis_impl,
    _logcdf_impl,
    _logpdf_impl,
    _mean_impl,
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
from ...dist_math.logistic import (
    logistic_cdf,
    logistic_logcdf,
    logistic_logpdf,
    logistic_pdf,
    logistic_ppf,
    logistic_sf,
)
from ...random_utils import split_tree
from ...tree_utils import tree_map_dist_at, zeros_pytree
from .._base import ContinuousUnivariateDistribution


class Logistic(ContinuousUnivariateDistribution):
    """Logistic distribution.
        X ~ Logistic(μ, σ)

    Args:
        loc (ArrayLike): Loc of the distribution.
        scale (ArrayLike): Scale of the distribution.
        dtype (jax.numpy.dtype): dtype of the distribution, default jnp.float_.
        use_batch (bool): Whether to use with vmap. Default False.
        
    Examples:
        >>> import jax.numpy as jnp
        >>> from fenbux import logpdf
        >>> from fenbux.univariate import Logistic
        >>> dist = Logistic(1.0, 1.0)
        >>> logpdf(dist, jnp.ones((10, )))
    """

    loc: PyTreeVar
    scale: PyTreeVar

    def __init__(self, loc=0.0, scale=1.0, dtype=jnp.float_, use_batch=False):
        _check_params_equal_tree_strcutre(loc, scale, use_batch=use_batch)
        self.loc, self.scale = _intialize_params_tree(
            loc, scale, use_batch=use_batch, dtype=dtype
        )


@_params_impl.dispatch
def params(dist: Logistic):
    dist = dist.broadcast_params()
    return dist.loc, dist.scale


@_support_impl.dispatch
def support(dist: Logistic):
    dist = dist.broadcast_params()
    return jtu.tree_map(lambda m: jnp.zeros_like(m), dist.loc), jtu.tree_map(
        lambda m: jnp.full_like(m, jnp.inf), dist.loc
    )


@_mean_impl.dispatch
def mean(dist: Logistic):
    dist = dist.broadcast_params()
    return dist.loc


@_variance_impl.dispatch
def variance(dist: Logistic):
    dist = dist.broadcast_params()
    return jtu.tree_map(lambda s: jnp.pi**2 * s**2 / 3, dist.scale)


@_standard_dev_impl.dispatch
def standard_dev(dist: Logistic):
    dist = dist.broadcast_params()
    return jtu.tree_map(lambda s: jnp.pi * s / jnp.sqrt(3), dist.scale)


@_skewness_impl.dispatch
def skewness(dist: Logistic):
    dist = dist.broadcast_params()
    return zeros_pytree(dist.loc)


@_kurtosis_impl.dispatch
def kurtosis(dist: Logistic):
    dist = dist.broadcast_params()
    return jtu.tree_map(lambda s: 6 / 5, dist.scale)


@_entropy_impl.dispatch
def entropy(dist: Logistic):
    dist = dist.broadcast_params()
    entropy = jtu.tree_map(lambda s: jnp.log(s) + 2, dist.scale)
    return entropy


@_logpdf_impl.dispatch
def logpdf(dist: Logistic, x: ArrayLike):
    dist = dist.broadcast_params()
    return tree_map_dist_at(logistic_logpdf, dist, x)


@_pdf_impl.dispatch
def pdf(dist: Logistic, x: ArrayLike):
    dist = dist.broadcast_params()
    return tree_map_dist_at(logistic_pdf, dist, x)


@_cdf_impl.dispatch
def cdf(dist: Logistic, x: ArrayLike):
    dist = dist.broadcast_params()
    return tree_map_dist_at(logistic_cdf, dist, x)


@_logcdf_impl.dispatch
def logcdf(dist: Logistic, x: ArrayLike):
    dist = dist.broadcast_params()
    return tree_map_dist_at(logistic_logcdf, dist, x)


@_sf_impl.dispatch
def sf(dist: Logistic, x: ArrayLike):
    dist = dist.broadcast_params()
    return tree_map_dist_at(logistic_sf, dist, x)


@_quantile_impl.dispatch
def quantile(dist: Logistic, x: ArrayLike):
    dist = dist.broadcast_params()
    return tree_map_dist_at(logistic_ppf, dist, x)


@_rand_impl.dispatch
def rand(
    dist: Logistic, key: KeyArray, shape: Shape = (), dtype: DTypeLikeFloat = float
):
    dist = dist.broadcast_params()
    _key_tree = split_tree(key, dist.loc)
    rvs = jtu.tree_map(
        lambda l, s, k: jr.logistic(k, shape=shape, dtype=dtype) * s + l,
        dist.loc,
        dist.scale,
        _key_tree,
    )
    return rvs