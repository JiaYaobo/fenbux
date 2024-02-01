import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import ArrayLike

from ...core import (
    _cdf_impl,
    _check_params_equal_tree_strcutre,
    _intialize_params_tree,
    _logcdf_impl,
    _logpdf_impl,
    _params_impl,
    _pdf_impl,
    _quantile_impl,
    _rand_impl,
    _sf_impl,
    _support_impl,
    DTypeLikeFloat,
    KeyArray,
    PyTreeVar,
    Shape,
)
from ...dist_math.cauchy import (
    cauchy_cdf,
    cauchy_logcdf,
    cauchy_logpdf,
    cauchy_pdf,
    cauchy_ppf,
    cauchy_sf,
)
from ...random_utils import split_tree
from ...tree_utils import tree_map_dist_at
from .._base import ContinuousUnivariateDistribution


class Cauchy(ContinuousUnivariateDistribution):
    """Cauchy distribution.

    Args:
        loc (PyTreeVar): Location parameter of the distribution.
        scale (PyTreeVar): Scale parameter of the distribution.
        dtype (jax.numpy.dtype): dtype of the distribution, default jnp.float_.
        use_batch (bool): Whether to use with vmap. Default False.

    Examples:
        >>> import jax.numpy as jnp
        >>> from fenbux import logpdf
        >>> from fenbux.univariate import Cauchy
        >>> dist = Cauchy(1.0, 1.0)
        >>> logpdf(dist, jnp.ones((10, )))
    """

    loc: PyTreeVar
    scale: PyTreeVar

    def __init__(
        self,
        loc: PyTreeVar,
        scale: PyTreeVar,
        dtype: DTypeLikeFloat = jnp.float_,
        use_batch: bool = False,
    ):
        _check_params_equal_tree_strcutre(loc, scale, use_batch=use_batch)
        self.loc, self.scale = _intialize_params_tree(
            loc, scale, use_batch=use_batch, dtype=dtype
        )


@_params_impl.dispatch
def _params_impl(dist: Cauchy):
    return (dist.loc, dist.scale)


@_support_impl.dispatch
def _support_impl(dist: Cauchy):
    dist = dist.broadcast_params()
    return (
        jtu.tree_map(lambda _: -jnp.inf, dist.loc),
        jtu.tree_map(lambda _: jnp.inf, dist.loc),
    )


@_logpdf_impl.dispatch
def _logpdf_impl(dist: Cauchy, x: ArrayLike):
    dist = dist.broadcast_params()
    return tree_map_dist_at(cauchy_logpdf, dist, x)


@_pdf_impl.dispatch
def _pdf_impl(dist: Cauchy, x: ArrayLike):
    dist = dist.broadcast_params()
    return tree_map_dist_at(cauchy_pdf, dist, x)


@_cdf_impl.dispatch
def _cdf_impl(dist: Cauchy, x: ArrayLike):
    dist = dist.broadcast_params()
    return tree_map_dist_at(cauchy_cdf, dist, x)


@_logcdf_impl.dispatch
def _logcdf_impl(dist: Cauchy, x: ArrayLike):
    dist = dist.broadcast_params()
    return tree_map_dist_at(cauchy_logcdf, dist, x)


@_quantile_impl.dispatch
def _quantile_impl(dist: Cauchy, x: ArrayLike):
    dist = dist.broadcast_params()
    return tree_map_dist_at(cauchy_ppf, dist, x)


@_sf_impl.dispatch
def _sf_impl(dist: Cauchy, x: ArrayLike):
    dist = dist.broadcast_params()
    return tree_map_dist_at(cauchy_sf, dist, x)


@_rand_impl.dispatch
def _rand_impl(
    key: KeyArray, shape: Shape, dist: Cauchy, dtype: DTypeLikeFloat = jnp.float_
):
    dist = dist.broadcast_params()
    _key_tree = split_tree(key, dist.loc)
    rvs = jtu.tree_map(
        lambda loc, scale, key: jr.cauchy(key, shape, dtype=dtype) * scale + loc,
        dist.loc,
        dist.scale,
        _key_tree,
    )
    return rvs
