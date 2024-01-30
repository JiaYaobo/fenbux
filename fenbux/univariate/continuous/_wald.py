import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import ArrayLike

from ...core import (
    _cdf_impl,
    _check_params_equal_tree_strcutre,
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
from ...dist_math.wald import (
    wald_cdf,
    wald_logcdf,
    wald_logpdf,
    wald_pdf,
    wald_ppf,
    wald_sf,
)
from ...random_utils import split_tree
from ...tree_utils import tree_map_dist_at
from .._base import ContinuousUnivariateDistribution


class Wald(ContinuousUnivariateDistribution):
    """Wald distribution.

        X ~ Wald(mu)

    Args:
        mu (PyTree): Mean parameter of the distribution.
        dtype (jax.numpy.dtype): dtype of the distribution, default jnp.float_.
        use_batch (bool): Whether to use with vmap. Default False.

    Examples:
        >>> import jax.numpy as jnp
        >>> from fenbux import logpdf
        >>> from fenbux.univariate import Wald
        >>> dist = Wald(1.0)
        >>> logpdf(dist, jnp.ones((10, )))
    """

    mu: PyTreeVar

    def __init__(
        self,
        mu: PyTreeVar,
        dtype: DTypeLikeFloat = jnp.float_,
        use_batch: bool = False,
    ):
        _check_params_equal_tree_strcutre(mu, use_batch=use_batch)
        self.mu = _intialize_params_tree(mu, use_batch=use_batch, dtype=dtype)


@_params_impl.dispatch
def _params_impl(dist: Wald):
    return (dist.mu, )


@_support_impl.dispatch
def _support_impl(dist: Wald):
    dist = dist.broadcast_params()
    return jtu.tree_map(lambda m: jnp.zeros_like(m), dist.mu), jtu.tree_map(
        lambda m: jnp.full_like(m, jnp.inf), dist.mu
    )


@_mean_impl.dispatch
def _mean_impl(dist: Wald):
    dist = dist.broadcast_params()
    return dist.mu


@_variance_impl.dispatch
def _variance_impl(dist: Wald):
    dist = dist.broadcast_params()
    return jtu.tree_map(lambda m: m**3, dist.mu)


@_standard_dev_impl.dispatch
def _standard_dev_impl(dist: Wald):
    dist = dist.broadcast_params()
    return jtu.tree_map(lambda m: m**1.5, dist.mu)


@_skewness_impl.dispatch
def _skewness_impl(dist: Wald):
    dist = dist.broadcast_params()
    return jtu.tree_map(lambda m: 3 * (m) ** 0.5, dist.mu)


@_kurtosis_impl.dispatch
def _kurtosis_impl(dist: Wald):
    dist = dist.broadcast_params()
    return jtu.tree_map(lambda m: 15 * m, dist.mu)


@_logpdf_impl.dispatch
def _logpdf_impl(dist: Wald, x: ArrayLike):
    dist = dist.broadcast_params()
    return tree_map_dist_at(wald_logpdf, dist, x)


@_pdf_impl.dispatch
def _pdf_impl(dist: Wald, x: ArrayLike):
    dist = dist.broadcast_params()
    return tree_map_dist_at(wald_pdf, dist, x)


@_logcdf_impl.dispatch
def _logcdf_impl(dist: Wald, x: ArrayLike):
    dist = dist.broadcast_params()
    return tree_map_dist_at(wald_logcdf, dist, x)


@_cdf_impl.dispatch
def _cdf_impl(dist: Wald, x: ArrayLike):
    dist = dist.broadcast_params()
    return tree_map_dist_at(wald_cdf, dist, x)


@_sf_impl.dispatch
def _sf_impl(dist: Wald, x: ArrayLike):
    dist = dist.broadcast_params()
    return tree_map_dist_at(wald_sf, dist, x)


@_quantile_impl.dispatch
def _quantile_impl(dist: Wald, x: ArrayLike):
    dist = dist.broadcast_params()
    return tree_map_dist_at(wald_ppf, dist, x)


@_rand_impl.dispatch
def _rand(
    d: Wald, key: KeyArray, shape: Shape = (), dtype: DTypeLikeFloat = float
):
    d = d.broadcast_params()
    _key_tree = split_tree(key, d.mu)
    return jtu.tree_map(
        lambda m, key: jr.wald(key, mean=m, shape=shape, dtype=dtype),
        d.mu,
        _key_tree,
    )
