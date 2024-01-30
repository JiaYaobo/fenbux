import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jax import lax
from jaxtyping import ArrayLike

from ...core import (
    _cdf_impl,
    _cf_impl,
    _entropy_impl,
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
from ...dist_math.bernoulli import (
    bernoulli_cdf,
    bernoulli_cf,
    bernoulli_logcdf,
    bernoulli_logpmf,
    bernoulli_mgf,
    bernoulli_pmf,
    bernoulli_ppf,
    bernoulli_sf,
)
from ...random_utils import split_tree
from ...tree_utils import tree_map_dist_at
from .._base import DiscreteUnivariateDistribution


class Bernoulli(DiscreteUnivariateDistribution):
    """Bernoulli distribution.
        X ~ Bernoulli(p)
    Args:
        p (PyTree): Probability of success.
        dtype (jax.numpy.dtype): dtype of the distribution, default jnp.float_.
        use_batch (bool): Whether to use with vmap. Default False.

    Examples:
        >>> import jax.numpy as jnp
        >>> from fenbux import logpdf
        >>> from fenbux.univariate import Bernoulli
        >>> dist = Bernoulli(0.5)
        >>> logpdf(dist, jnp.ones((10, )))
    """

    p: PyTreeVar

    def __init__(self, p=0.0, dtype=jnp.float_, use_batch=False):
        self.p = _intialize_params_tree(p, use_batch=use_batch, dtype=dtype)


@_params_impl.dispatch
def _params(d: Bernoulli):
    return (d.p,)


@_support_impl.dispatch
def _support(d: Bernoulli):
    return jtu.tree_map(lambda p: jnp.zeros_like(p), d.p), jtu.tree_map(
        lambda p: jnp.ones_like(p), d.p
    )


@_mean_impl.dispatch
def _mean(d: Bernoulli):
    return jtu.tree_map(lambda p: p, d.p)


@_variance_impl.dispatch
def _variance(d: Bernoulli):
    return jtu.tree_map(lambda p: p * (1 - p), d.p)


@_standard_dev_impl.dispatch
def _standard_dev(d: Bernoulli):
    return jtu.tree_map(lambda p: jnp.sqrt(p * (1 - p)), d.p)


@_kurtosis_impl.dispatch
def _kurtosis(d: Bernoulli):
    return jtu.tree_map(
        lambda p: jnp.where(
            lax.gt(p, 0.0) & lax.lt(p, 1.0),
            (1 - 6 * p * (1 - p)) / (p * (1 - p)),
            jnp.nan,
        ),
        d.p,
    )


@_skewness_impl.dispatch
def _skewness(d: Bernoulli):
    return jtu.tree_map(
        lambda p: jnp.where(
            lax.gt(p, 0.0) & lax.lt(p, 1.0),
            (1 - 2 * p) / jnp.sqrt(p * (1 - p)),
            jnp.nan,
        ),
        d.p,
    )


@_entropy_impl.dispatch
def _entropy(d: Bernoulli):
    return jtu.tree_map(lambda p: -p * jnp.log(p) - (1 - p) * jnp.log(1 - p), d.p)


@_pmf_impl.dispatch
def _pmf(d: Bernoulli, x: ArrayLike):
    return tree_map_dist_at(bernoulli_pmf, d, x)


@_logpmf_impl.dispatch
def _logpmf(d: Bernoulli, x: ArrayLike):
    return tree_map_dist_at(bernoulli_logpmf, d, x)


@_rand_impl.dispatch
def _rand(
    d: Bernoulli, key: KeyArray, shape: Shape = (), dtype: DTypeLikeFloat = float
):
    _key_tree = split_tree(key, d.p)
    rvs = jtu.tree_map(
        lambda p, k: jr.bernoulli(k, p, shape=shape).astype(dtype),
        d.p,
        _key_tree,
    )
    return rvs


@_cdf_impl.dispatch
def _cdf(d: Bernoulli, x: ArrayLike):
    return tree_map_dist_at(bernoulli_cdf, d, x)


@_logcdf_impl.dispatch
def _logcdf(d: Bernoulli, x: ArrayLike):
    return tree_map_dist_at(bernoulli_logcdf, d, x)


@_quantile_impl.dispatch
def _quantile(d: Bernoulli, x: ArrayLike):
    return tree_map_dist_at(bernoulli_ppf, d, x)


@_mgf_impl.dispatch
def _mgf(d: Bernoulli, t: ArrayLike):
    return tree_map_dist_at(bernoulli_mgf, d, t)


@_cf_impl.dispatch
def _cf(d: Bernoulli, t: ArrayLike):
    return tree_map_dist_at(bernoulli_cf, d, t)


@_sf_impl.dispatch
def _sf(d: Bernoulli, x: ArrayLike):
    return tree_map_dist_at(bernoulli_sf, d, x)
