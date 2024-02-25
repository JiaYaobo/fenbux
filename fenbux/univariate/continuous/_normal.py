import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import ArrayLike

from ...core import (
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
from ...dist_math.normal import (
    normal_cdf,
    normal_cf,
    normal_logcdf,
    normal_logpdf,
    normal_mgf,
    normal_pdf,
    normal_ppf,
    normal_sf,
)
from ...random_utils import split_tree
from ...tree_utils import tree_map_dist_at, zeros_pytree
from .._base import ContinuousUnivariateDistribution


class Normal(ContinuousUnivariateDistribution):
    """Normal distribution.

    Args:
        mean (ArrayLike): Mean of the distribution.
        sd (ArrayLike): Standard deviation of the distribution.
        dtype (jax.numpy.dtype): dtype of the distribution, default jnp.float_.
        use_batch (bool): Whether to use with vmap. Default False.

    Examples:
        >>> import jax.numpy as jnp
        >>> from jax import vmap
        >>> from fenbux import logpdf
        >>> from fenbux.univariate import Normal
        >>> dist = Normal(0.0, jnp.ones((10, )))
    """

    mean: PyTreeVar
    sd: PyTreeVar

    def __init__(self, mean=0.0, sd=0.0, dtype=jnp.float_, use_batch=False):
        _check_params_equal_tree_strcutre(mean, sd, use_batch=use_batch)
        self.mean, self.sd = _intialize_params_tree(
            mean, sd, use_batch=use_batch, dtype=dtype
        )


@_params_impl.dispatch
def _params(d: Normal):
    return (d.mean, d.sd)


@_support_impl.dispatch
def _domain(d: Normal):
    dist = d.broadcast_params()
    return jtu.tree_map(lambda m: jnp.full_like(m, -jnp.inf), dist.mean), jtu.tree_map(
        lambda m: jnp.full_like(m, jnp.inf), dist.mean
    )


@_mean_impl.dispatch
def _mean(d: Normal):
    return d.broadcast_params().mean


@_variance_impl.dispatch
def _variance(d: Normal):
    return jtu.tree_map(lambda x: x**2, d.broadcast_params().sd)


@_standard_dev_impl.dispatch
def _std(d: Normal):
    return d.broadcast_params().sd


@_kurtosis_impl.dispatch
def _kurtosis(d: Normal):
    shape = d.broadcast_shapes()
    return zeros_pytree(shape)


@_skewness_impl.dispatch
def _skewness(d: Normal):
    shape = d.broadcast_shapes()
    return zeros_pytree(shape)


@_entropy_impl.dispatch
def _entropy(d: Normal):
    d = d.broadcast_params()
    entropy = jtu.tree_map(lambda σ: 0.5 * jnp.log(2 * jnp.pi * σ**2) + 0.5, d.sd)
    return entropy


@_logpdf_impl.dispatch
def _logpdf(d: Normal, x: ArrayLike):
    d = d.broadcast_params()
    return tree_map_dist_at(normal_logpdf, d, x)


@_pdf_impl.dispatch
def _pdf(d: Normal, x: ArrayLike):
    d = d.broadcast_params()
    return tree_map_dist_at(normal_pdf, d, x)


@_logcdf_impl.dispatch
def _logcdf(d: Normal, x: ArrayLike):
    d = d.broadcast_params()
    return tree_map_dist_at(normal_logcdf, d, x)


@_cdf_impl.dispatch
def _cdf(d: Normal, x: ArrayLike):
    d = d.broadcast_params()
    return tree_map_dist_at(normal_cdf, d, x)


@_quantile_impl.dispatch
def _quantile(d: Normal, q: ArrayLike):
    d = d.broadcast_params()
    return tree_map_dist_at(normal_ppf, d, q)


@_mgf_impl.dispatch
def _mgf(d: Normal, t: ArrayLike):
    d = d.broadcast_params()
    return tree_map_dist_at(normal_mgf, d, t)


@_cf_impl.dispatch
def _cf(d: Normal, t: ArrayLike):
    d = d.broadcast_params()
    return tree_map_dist_at(normal_cf, d, t)


@_sf_impl.dispatch
def _sf(d: Normal, x: ArrayLike):
    d = d.broadcast_params()
    return tree_map_dist_at(normal_sf, d, x)


@_rand_impl.dispatch
def _rand(d: Normal, key: KeyArray, shape: Shape = (), dtype: DTypeLikeFloat = float):
    d = d.broadcast_params()
    _key_tree = split_tree(key, d.mean)
    rvs = jtu.tree_map(
        lambda μ, σ, key: jr.normal(key, shape, dtype=dtype) * σ + μ,
        d.mean,
        d.sd,
        _key_tree,
    )
    return rvs
