import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jax.scipy.special import gamma
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
from ...dist_math.weibull import (
    weibull_cdf,
    weibull_logcdf,
    weibull_logpdf,
    weibull_pdf,
    weibull_ppf,
    weibull_sf,
)
from ...random_utils import split_tree
from ...tree_utils import tree_map_dist_at
from .._base import ContinuousUnivariateDistribution


class Weibull(ContinuousUnivariateDistribution):
    """Weibull distribution.

        X ~ Weibull(shape, scale)

    Args:
        shape (PyTree): Shape parameter of the distribution.
        scale (PyTree): Scale parameter of the distribution.
        dtype (jax.numpy.dtype): dtype of the distribution, default jnp.float_.
        use_batch (bool): Whether to use with vmap. Default False.

    Examples:
        >>> import jax.numpy as jnp
        >>> from fenbux import logpdf
        >>> from fenbux.univariate import Weibull
        >>> dist = WeiBull(1.0, 1.0)
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
def _params(d: Weibull):
    return (d.shape, d.scale)


@_support_impl.dispatch
def _support(d: Weibull):
    d = d.broadcast_params()
    return jtu.tree_map(lambda s: jnp.zeros_like(s), d.shape), jtu.tree_map(
        lambda s: jnp.full_like(s, jnp.inf), d.shape
    )


@_mean_impl.dispatch
def _mean(d: Weibull):
    d = d.broadcast_params()
    return jtu.tree_map(
        lambda shape, scale: scale * gamma(1.0 + 1.0 / shape), d.shape, d.scale
    )


@_variance_impl.dispatch
def _variance(d: Weibull):
    d = d.broadcast_params()
    return jtu.tree_map(
        lambda shape, scale: scale**2 * gamma(1.0 + 2.0 / shape) - _mean_impl(d) ** 2,
        d.shape,
        d.scale,
    )


@_standard_dev_impl.dispatch
def _standard_dev(d: Weibull):
    d = d.broadcast_params()
    return jtu.tree_map(
        lambda shape, scale: scale
        * jnp.sqrt(gamma(1.0 + 2.0 / shape) - _mean_impl(d) ** 2),
        d.shape,
        d.scale,
    )


@_skewness_impl.dispatch
def _skewness(d: Weibull):
    d = d.broadcast_params()

    def _fn(shape):
        gamma1 = gamma(1.0 + 1.0 / shape)
        gamma2 = gamma(1.0 + 2.0 / shape)
        gamma3 = gamma(1.0 + 3.0 / shape)
        return (gamma3 - 3 * gamma2 * gamma1 + 2 * gamma1**3) / (
            gamma2 - gamma1**2
        ) ** 1.5

    return jtu.tree_map(
        lambda shape: _fn(shape),
        d.shape,
    )


@_kurtosis_impl.dispatch
def _kurtosis(d: Weibull):
    d = d.broadcast_params()

    def _fn(shape):
        gamma1 = gamma(1.0 + 1.0 / shape)
        gamma2 = gamma(1.0 + 2.0 / shape)
        gamma3 = gamma(1.0 + 3.0 / shape)
        gamma4 = gamma(1.0 + 4.0 / shape)
        return (
            gamma4 - 4 * gamma3 * gamma1 + 6 * gamma2 * gamma1**2 - 3 * gamma1**4
        ) / (gamma2 - gamma1**2) ** 2

    return jtu.tree_map(
        lambda shape: _fn(shape),
        d.shape,
    )


@_entropy_impl.dispatch
def _entropy(d: Weibull):
    d = d.broadcast_params()
    return jtu.tree_map(
        lambda shape, scale: (
            jnp.euler_gamma * (1.0 - 1.0 / shape) + jnp.log(scale / shape) + 1.0
        ),
        d.shape,
        d.scale,
    )


@_logpdf_impl.dispatch
def _logpdf(d: Weibull, x: ArrayLike):
    d = d.broadcast_params()
    return tree_map_dist_at(weibull_logpdf, d, x)


@_pdf_impl.dispatch
def _pdf(d: Weibull, x: ArrayLike):
    d = d.broadcast_params()
    return tree_map_dist_at(weibull_pdf, d, x)


@_cdf_impl.dispatch
def _cdf(d: Weibull, x: ArrayLike):
    d = d.broadcast_params()
    return tree_map_dist_at(weibull_cdf, d, x)


@_logcdf_impl.dispatch
def _logcdf(d: Weibull, x: ArrayLike):
    d = d.broadcast_params()
    return tree_map_dist_at(weibull_logcdf, d, x)


@_quantile_impl.dispatch
def _quantile(d: Weibull, x: ArrayLike):
    d = d.broadcast_params()
    return tree_map_dist_at(weibull_ppf, d, x)


@_sf_impl.dispatch
def _sf(d: Weibull, x: ArrayLike):
    d = d.broadcast_params()
    return tree_map_dist_at(weibull_sf, d, x)


@_rand_impl.dispatch
def _rand(
    d: Weibull, key: KeyArray, shape: Shape = (), dtype: DTypeLikeFloat = float
):
    d = d.broadcast_params()
    _key_tree = split_tree(key, d.shape)
    return jtu.tree_map(
        lambda shape_, scale, key: jr.weibull_min(
            key, scale=scale, concentration=shape_, shape=shape, dtype=dtype
        ),
        d.shape,
        d.scale,
        _key_tree,
    )
