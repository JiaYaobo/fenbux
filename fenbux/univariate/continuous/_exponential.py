import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import ArrayLike

from ...core import (
    _cdf_impl,
    _cf_impl,
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
from ...dist_math.exp import (
    exp_cdf,
    exp_cf,
    exp_logcdf,
    exp_logpdf,
    exp_mgf,
    exp_pdf,
    exp_ppf,
    exp_sf,
)
from ...random_utils import split_tree
from ...tree_utils import tree_map_dist_at
from .._base import ContinuousUnivariateDistribution


class Exponential(ContinuousUnivariateDistribution):
    """Exponential distribution.

    Args:
        rate (PyTree): Rate parameter.
        dtype (jax.numpy.dtype): dtype of the distribution, default jnp.float_.
        use_batch (bool): Whether to use with vmap. Default False.

    Examples:
        >>> import jax.numpy as jnp
        >>> from fenbux import logpdf
        >>> from fenbux.univariate import Exponential
        >>> dist = Exponential(1.0)
        >>> logpdf(dist, jnp.ones((10, )))
    """

    rate: PyTreeVar

    def __init__(
        self,
        rate: PyTreeVar = 1.0,
        dtype=jnp.float_,
        use_batch=False,
    ):
        self.rate = _intialize_params_tree(rate, use_batch=use_batch, dtype=dtype)


@_params_impl.dispatch
def _params(d: Exponential):
    return (d.rate,)


@_support_impl.dispatch
def _support(d: Exponential):
    return jtu.tree_map(lambda r: jnp.zeros_like(r), d.rate), jtu.tree_map(
        lambda r: jnp.full_like(r, jnp.inf), d.rate
    )


@_mean_impl.dispatch
def _mean(d: Exponential):
    return jtu.tree_map(lambda x: 1.0 / x, d.rate)


@_variance_impl.dispatch
def _variance(d: Exponential):
    return jtu.tree_map(lambda x: 1.0 / x**2, d.rate)


@_standard_dev_impl.dispatch
def _standard_dev(d: Exponential):
    return jtu.tree_map(lambda x: 1.0 / x, d.rate)


@_skewness_impl.dispatch
def _skewness(d: Exponential):
    return jtu.tree_map(lambda x: 2.0, d.rate)


@_kurtosis_impl.dispatch
def _kurtosis(d: Exponential):
    return jtu.tree_map(lambda x: 6.0, d.rate)


@_support_impl.dispatch
def _support(d: Exponential):
    return jtu.tree_map(lambda x: (0.0, jnp.inf), d.rate)


@_entropy_impl.dispatch
def _entropy(d: Exponential):
    return jtu.tree_map(lambda x: 1.0 - jnp.log(x), d.rate)


@_logpdf_impl.dispatch
def _logpdf(d: Exponential, x: ArrayLike):
    return tree_map_dist_at(exp_logpdf, d, x)


@_pdf_impl.dispatch
def _pdf(d: Exponential, x: ArrayLike):
    return tree_map_dist_at(exp_pdf, d, x)


@_logcdf_impl.dispatch
def _logcdf(d: Exponential, x: ArrayLike):
    return tree_map_dist_at(exp_logcdf, d, x)


@_cdf_impl.dispatch
def _cdf(d: Exponential, x: ArrayLike):
    return tree_map_dist_at(exp_cdf, d, x)


@_quantile_impl.dispatch
def _quantile(d: Exponential, x: ArrayLike):
    return tree_map_dist_at(exp_ppf, d, x)


@_sf_impl.dispatch
def _sf(d: Exponential, x: ArrayLike):
    return tree_map_dist_at(exp_sf, d, x)


@_mgf_impl.dispatch
def _mgf(d: Exponential, t: ArrayLike):
    return tree_map_dist_at(exp_mgf, d, t)


@_cf_impl.dispatch
def _cf(d: Exponential, t: ArrayLike):
    return tree_map_dist_at(exp_cf, d, t)


@_rand_impl.dispatch
def _rand(
    d: Exponential, key: KeyArray, shape: Shape = (), dtype: DTypeLikeFloat = float
):
    _key_tree = split_tree(key, d.rate)
    return jtu.tree_map(
        lambda r, k: jr.exponential(k, shape, dtype) / r, d.rate, _key_tree
    )
