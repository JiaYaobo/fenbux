import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

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
from ...dist_math.pareto import (
    pareto_cdf,
    pareto_logcdf,
    pareto_logpdf,
    pareto_pdf,
    pareto_ppf,
    pareto_sf,
)
from ...random_utils import split_tree
from ...tree_utils import tree_map_dist_at
from .._base import ContinuousUnivariateDistribution


class Pareto(ContinuousUnivariateDistribution):
    """Pareto distribution.

        X ~ Pareto(shape, scale)

    Args:
        shape (PyTree): Shape parameter of the distribution.
        scale (PyTree): Scale parameter of the distribution.
        dtype (jax.numpy.dtype): dtype of the distribution, default jnp.float_.
        use_batch (bool): Whether to use with vmap. Default False.

    Examples:
        >>> import jax.numpy as jnp
        >>> from fenbux import logpdf
        >>> from fenbux.univariate import Pareto
        >>> dist = Pareto(1.0, 1.0)
        >>> logpdf(dist, jnp.ones((10, )))
    """

    shape: PyTreeVar
    scale: PyTreeVar

    def __init__(self, shape=0.0, scale=0.0, dtype=jnp.float_, use_batch=False):
        _check_params_equal_tree_strcutre(shape, scale, use_batch=use_batch)
        self.shape, self.scale = _intialize_params_tree(
            shape, scale, use_batch=use_batch, dtype=dtype
        )


@_params_impl.dispatch
def _params(d: Pareto):
    return (d.shape, d.scale)


@_support_impl.dispatch
def _support(d: Pareto):
    d = d.broadcast_params()
    return jtu.tree_map(lambda scale: scale, d.scale), jtu.tree_map(
        lambda scale: jnp.full_like(scale, jnp.inf), d.scale
    )


@_mean_impl.dispatch
def _mean(d: Pareto):
    d = d.broadcast_params()
    return jtu.tree_map(
        lambda a, b: jnp.where(a > 1.0, b * a / (a - 1.0), jnp.inf), d.shape, d.scale
    )


@_variance_impl.dispatch
def _variance(d: Pareto):
    d = d.broadcast_params()
    return jtu.tree_map(
        lambda a, b: jnp.where(
            a > 2.0, b**2 * a / ((a - 1.0) ** 2 * (a - 2.0)), jnp.inf
        ),
        d.shape,
        d.scale,
    )


@_standard_dev_impl.dispatch
def _standard_dev(d: Pareto):
    d = d.broadcast_params()
    return jtu.tree_map(
        lambda a, b: jnp.where(
            a > 2.0, b * jnp.sqrt(a) / ((a - 1.0) * jnp.sqrt(a - 2.0)), jnp.inf
        ),
        d.shape,
        d.scale,
    )


@_skewness_impl.dispatch
def _skewness(d: Pareto):
    d = d.broadcast_params()
    return jtu.tree_map(
        lambda a: jnp.where(
            a > 3, 2.0 * (1.0 + a) / (a - 3.0) * jnp.sqrt((a - 2.0) / a), jnp.nan
        ),
        d.shape,
    )


@_kurtosis_impl.dispatch
def _kurtosis(d: Pareto):
    d = d.broadcast_params()
    return jtu.tree_map(
        lambda a: jnp.where(
            a > 4,
            6.0 * (a**3 + a**2 - 6.0 * a - 2.0) / (a * (a - 3.0) * (a - 4.0)),
            jnp.nan,
        ),
        d.shape,
    )


@_entropy_impl.dispatch
def _entropy(d: Pareto):
    d = d.broadcast_params()
    return jtu.tree_map(
        lambda a, b: jnp.log(b) - jnp.log(a) + 1.0 / a + 1.0, d.shape, d.scale
    )


@_logpdf_impl.dispatch
def _logpdf(d: Pareto, x):
    d = d.broadcast_params()
    return tree_map_dist_at(pareto_logpdf, d, x)


@_pdf_impl.dispatch
def _pdf(d: Pareto, x):
    d = d.broadcast_params()
    return tree_map_dist_at(pareto_pdf, d, x)


@_logcdf_impl.dispatch
def _logcdf(d: Pareto, x):
    d = d.broadcast_params()
    return tree_map_dist_at(pareto_logcdf, d, x)


@_cdf_impl.dispatch
def _cdf(d: Pareto, x):
    d = d.broadcast_params()
    return tree_map_dist_at(pareto_cdf, d, x)


@_quantile_impl.dispatch
def _quantile(d: Pareto, x):
    d = d.broadcast_params()
    return tree_map_dist_at(pareto_ppf, d, x)


@_sf_impl.dispatch
def _sf(d: Pareto, x):
    d = d.broadcast_params()
    return tree_map_dist_at(pareto_sf, d, x)


@_rand_impl.dispatch
def _rand(d: Pareto, key: KeyArray, shape: Shape = (), dtype: DTypeLikeFloat = float):
    d = d.broadcast_params()
    _key_tree = split_tree(key, d.shape)
    return jtu.tree_map(
        lambda a, b, k: jr.pareto(k, a, shape, dtype) * b, d.shape, d.scale, _key_tree
    )
