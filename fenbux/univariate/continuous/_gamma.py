import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import ArrayLike

from ...core import (
    _cdf_impl,
    _cf_impl,
    _check_params_equal_tree_strcutre,
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
from ...dist_math.gamma import (
    gamma_cdf,
    gamma_cf,
    gamma_logcdf,
    gamma_logpdf,
    gamma_mgf,
    gamma_pdf,
    gamma_ppf,
    gamma_sf,
)
from ...random_utils import split_tree
from ...tree_utils import tree_map_dist_at
from .._base import ContinuousUnivariateDistribution


class Gamma(ContinuousUnivariateDistribution):
    """Gamma distribution.

        X ~ Gamma(shape, rate)

    Args:
        shape (PyTree): Shape parameter of the distribution.
        rate (PyTree): Rate parameter of the distribution.
        dtype (jax.numpy.dtype): dtype of the distribution, default jnp.float_.
        use_batch (bool): Whether to use with vmap. Default False.

    Examples:
        >>> import jax.numpy as jnp
        >>> from fenbux import logpdf
        >>> from fenbux.univariate import Gamma
        >>> dist = Gamma(1.0, 1.0)
        >>> logpdf(dist, jnp.ones((10, )))
    """

    shape: PyTreeVar
    rate: PyTreeVar

    def __init__(self, shape=0.0, rate=0.0, dtype=jnp.float_, use_batch=False):
        _check_params_equal_tree_strcutre(shape, rate, use_batch=use_batch)
        self.shape, self.rate = _intialize_params_tree(
            shape, rate, use_batch=use_batch, dtype=dtype
        )


@_params_impl.dispatch
def _params(d: Gamma):
    return (d.shape, d.rate)


@_support_impl.dispatch
def _support(d: Gamma):
    d = d.broadcast_params().shape
    return jtu.tree_map(lambda r: jnp.zeros_like(r), d.rate), jtu.tree_map(
        lambda r: jnp.full_like(r, jnp.inf), d.rate
    )


@_mean_impl.dispatch
def _mean(d: Gamma):
    d = d.broadcast_params()
    return jtu.tree_map(lambda α, β: α / β, d.shape, d.rate)


@_variance_impl.dispatch
def _variance(d: Gamma):
    d = d.broadcast_params()
    return jtu.tree_map(lambda α, β: α / (β**2), d.shape, d.rate)


@_standard_dev_impl.dispatch
def _std(d: Gamma):
    d = d.broadcast_params()
    return jtu.tree_map(lambda α, β: jnp.sqrt(α / (β**2)), d.shape, d.rate)


@_kurtosis_impl.dispatch
def _kurtosis(d: Gamma):
    d = d.broadcast_params()
    return jtu.tree_map(lambda α: 6 / α, d.shape)


@_skewness_impl.dispatch
def _skewness(d: Gamma):
    d = d.broadcast_params()
    return jtu.tree_map(lambda α: 2 / jnp.sqrt(α), d.shape)


@_logpdf_impl.dispatch
def _logpdf(d: Gamma, x: ArrayLike):
    d = d.broadcast_params()
    return tree_map_dist_at(gamma_logpdf, d, x)


@_pdf_impl.dispatch
def _pdf(d: Gamma, x: ArrayLike):
    d = d.broadcast_params()
    return tree_map_dist_at(gamma_pdf, d, x)


@_logcdf_impl.dispatch
def _logcdf(d: Gamma, x: ArrayLike):
    d = d.broadcast_params()
    return tree_map_dist_at(gamma_logcdf, d, x)


@_cdf_impl.dispatch
def _cdf(d: Gamma, x: ArrayLike):
    d = d.broadcast_params()
    return tree_map_dist_at(gamma_cdf, d, x)


@_quantile_impl.dispatch
def _quantile(d: Gamma, q: ArrayLike):
    d = d.broadcast_params()
    return tree_map_dist_at(gamma_ppf, d, q)


@_mgf_impl.dispatch
def _mgf(d: Gamma, t: ArrayLike):
    d = d.broadcast_params()
    return tree_map_dist_at(gamma_mgf, d, t)


@_cf_impl.dispatch
def _cf(d: Gamma, t: ArrayLike):
    d = d.broadcast_params()
    return tree_map_dist_at(gamma_cf, d, t)


@_sf_impl.dispatch
def _sf(d: Gamma, x: ArrayLike):
    d = d.broadcast_params()
    return tree_map_dist_at(gamma_sf, d, x)


@_rand_impl.dispatch
def _rand(
    d: Gamma, key: KeyArray, shape: Shape = (), dtype: DTypeLikeFloat = float
):
    d = d.broadcast_params()
    _key_tree = split_tree(key, d.shape)
    rvs = jtu.tree_map(
        lambda α, β, k: jr.gamma(k, α, shape, dtype=dtype) * β,
        d.shape,
        d.rate,
        _key_tree,
    )
    return rvs


