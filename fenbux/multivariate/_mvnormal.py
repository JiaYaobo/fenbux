import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import ArrayLike

from ..core import (
    _cf_impl,
    _check_params_equal_tree_strcutre,
    _entropy_impl,
    _intialize_params_tree,
    _kurtosis_impl,
    _logpdf_impl,
    _mean_impl,
    _mgf_impl,
    _params_impl,
    _pdf_impl,
    _rand_impl,
    _skewness_impl,
    _standard_dev_impl,
    _support_impl,
    _variance_impl,
    DTypeLikeFloat,
    KeyArray,
    PyTreeVar,
    Shape,
)
from ..dist_math.mvnormal import (
    mvnormal_cf,
    mvnormal_logpdf,
    mvnormal_mgf,
    mvnormal_pdf,
)
from ..random_utils import split_tree
from ..tree_utils import (
    _is_multivariate_dist_params,
    tree_map_dist_at,
    zeros_like_pytree,
)
from ._base import ContinuousMultivariateDistribution


class MultivariateNormal(ContinuousMultivariateDistribution):
    """Multivariate normal distribution.
        X ~ Normal(μ, Σ)
    Args:
        mean (PyTree): Mean of the distribution.
        cov (PyTree): Covariance matrix of the distribution.
        dtype (jax.numpy.dtype): dtype of the distribution, default jnp.float_.
        use_batch (bool): Whether to use with vmap. Default False.

    Examples:
        >>> import jax.numpy as jnp
        >>> from fenbux import logpdf
        >>> from fenbux.distributions import MultivariateNormal
        >>> dist = MultivariateNormal(jnp.zeros((10, )), jnp.eye(10))
        >>> logpdf(dist, jnp.zeros((10, )))

    Attributes:
        mean (PyTree): Mean of the distribution.
        cov (PyTree): Covariance matrix of the distribution.
    """

    mean: PyTreeVar
    cov: PyTreeVar

    def __init__(self, mean=0.0, cov=0.0, dtype=jnp.float_, use_batch=False):
        _check_params_equal_tree_strcutre(mean, cov)
        self.mean, self.cov = _intialize_params_tree(
            mean, cov, use_batch=use_batch, dtype=dtype
        )


@_params_impl.dispatch
def _params_impl(dist: MultivariateNormal):
    return dist.mean, dist.cov


@_support_impl.dispatch
def _supp(d: MultivariateNormal):
    return jtu.tree_map(lambda m: jnp.full_like(m, -jnp.inf), d.mean), jtu.tree_map(
        lambda m: jnp.full_like(m, jnp.inf), d.mean
    )


@_mean_impl.dispatch
def _mean(d: MultivariateNormal):
    return (d.mean,)


@_variance_impl.dispatch
def _variance(d: MultivariateNormal):
    return d.cov


@_standard_dev_impl.dispatch
def _standard_dev(d: MultivariateNormal):
    return jtu.tree_map(
        lambda cov: jnp.sqrt(jnp.diag(cov)), d.cov, is_leaf=_is_multivariate_dist_params
    )


@_skewness_impl.dispatch
def _skewness(d: MultivariateNormal):
    return zeros_like_pytree(d.mean, is_leaf=_is_multivariate_dist_params)


@_kurtosis_impl.dispatch
def _kurtosis(d: MultivariateNormal):
    return zeros_like_pytree(d.mean, is_leaf=_is_multivariate_dist_params)


@_entropy_impl.dispatch
def _entropy(d: MultivariateNormal):
    return jtu.tree_map(
        lambda cov: 0.5 * jnp.log(jnp.linalg.det(2 * jnp.pi * jnp.e * cov)),
        d.cov,
        is_leaf=_is_multivariate_dist_params,
    )


@_logpdf_impl.dispatch
def _logpdf(d: MultivariateNormal, x: ArrayLike):
    return tree_map_dist_at(
        mvnormal_logpdf, d, x, is_leaf_dist=_is_multivariate_dist_params
    )


@_pdf_impl.dispatch
def _pdf(d: MultivariateNormal, x: ArrayLike):
    return tree_map_dist_at(
        mvnormal_pdf, d, x, is_leaf_dist=_is_multivariate_dist_params
    )


@_mgf_impl.dispatch
def _mgf(d: MultivariateNormal, t: ArrayLike):
    return tree_map_dist_at(
        mvnormal_mgf, d, t, is_leaf_dist=_is_multivariate_dist_params
    )


@_cf_impl.dispatch
def _cf(d: MultivariateNormal, t: ArrayLike):
    return tree_map_dist_at(
        mvnormal_cf, d, t, is_leaf_dist=_is_multivariate_dist_params
    )


@_rand_impl.dispatch
def _rand(
    d: MultivariateNormal, key: KeyArray, shape: Shape, dtype: DTypeLikeFloat = float
):
    _key_tree = split_tree(key, d.mean, is_leaf=_is_multivariate_dist_params)
    return jtu.tree_map(
        lambda mu, cov, k: jr.multivariate_normal(k, mu, cov, shape, dtype=dtype),
        d.mean,
        d.cov,
        _key_tree,
        is_leaf=lambda x: _is_multivariate_dist_params(x) or isinstance(x, KeyArray),
    )
