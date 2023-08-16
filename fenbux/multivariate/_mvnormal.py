import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

from ..core import (
    _cdf_impl,
    _cf_impl,
    _check_params_equal_tree_strcutre,
    _intialize_params_tree,
    _logcdf_impl,
    _logpdf_impl,
    _mgf_impl,
    _pdf_impl,
    _quantile_impl,
    _sf_impl,
    DTypeLikeFloat,
    _entropy_impl,
    KeyArray,
    _kurtosis_impl,
    _mean_impl,
    _params_impl,
    PyTreeVar,
    rand,
    Shape,
    _skewness_impl,
    _standard_dev_impl,
    _support_impl,
    _variance_impl,
)
from ..random_utils import split_tree
from ..tree_utils import zeros_pytree
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
        >>> from fenbux import MultivariateNormal, logpdf
        >>> dist = MultivariateNormal(jnp.zeros((10, )), jnp.eye(10))
        >>> # use vmap
        >>> vmap(logpdf, in_axes=(MultivariateNormal(None, 0, use_batch=True), 0))(dist, jnp.zeros((10, )))

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
def _domain(d: MultivariateNormal):
    return jtu.tree_map(lambda _: (-jnp.inf, jnp.inf), d.mean)


@_mean_impl.dispatch
def _mean(d: MultivariateNormal):
    return d.mean


@_variance_impl.dispatch
def _variance(d: MultivariateNormal):
    return d.cov


@_standard_dev_impl.dispatch
def _standard_dev(d: MultivariateNormal):
    return jnp.sqrt(d.cov)


@_skewness_impl.dispatch
def _skewness(d: MultivariateNormal):
    return zeros_pytree(d.mean)


@_kurtosis_impl.dispatch
def _kurtosis(d: MultivariateNormal):
    return zeros_pytree(d.mean)


@_entropy_impl.dispatch
def _entropy(d: MultivariateNormal):
    return 0.5 * jnp.log(jnp.linalg.det(2 * jnp.pi * jnp.e * d.cov))


@_logpdf_impl.dispatch
def _logpdf(d: MultivariateNormal, x):
    return jtu.tree_map(lambda mu, cov: _mvnormal_logpdf(x, mu, cov), d.mean, d.cov)


@_pdf_impl.dispatch
def _pdf(d: MultivariateNormal, x):
    return jtu.tree_map(lambda mu, cov: _mvnormal_pdf(x, mu, cov), d.mean, d.cov)


@_mgf_impl.dispatch
def _mgf(d: MultivariateNormal, t):
    return jtu.tree_map(lambda mu, cov: _mvnormal_mgf(t, mu, cov), d.mean, d.cov)


@_cf_impl.dispatch
def _cf(d: MultivariateNormal, t):
    return jtu.tree_map(lambda mu, cov: _mvnormal_cf(t, mu, cov), d.mean, d.cov)


@rand.dispatch
def _rand(d: MultivariateNormal, key: KeyArray, shape: Shape, dtype: DTypeLikeFloat):
    _key_tree = split_tree(key, d.mean)
    return jtu.tree_map(
        lambda mu, cov, k: jr.multivariate_normal(k, mu, cov, shape, dtype=dtype),
        d.mean,
        d.cov,
        _key_tree,
    )


def _mvnormal_pdf(x, mu, cov):
    def _fn(x, mu, cov):
        d = jnp.shape(x)[-1]
        x = x - mu
        cov_inv = jnp.linalg.inv(cov)
        return jnp.exp(
            -0.5 * jnp.einsum("...i,...ij,...j->...", x, cov_inv, x)
        ) / jnp.sqrt((2 * jnp.pi) ** d * jnp.linalg.det(cov))

    return jtu.tree_map(lambda xx: _fn(xx, mu, cov), x)


def _mvnormal_logpdf(x, mu, cov):
    def _fn(x, mu, cov):
        d = jnp.shape(x)[-1]
        x = x - mu
        cov_inv = jnp.linalg.inv(cov)
        return -0.5 * jnp.einsum("...i,...ij,...j->...", x, cov_inv, x) - 0.5 * (
            d * jnp.log(2 * jnp.pi) + jnp.log(jnp.linalg.det(cov))
        )

    return jtu.tree_map(lambda xx: _fn(xx, mu, cov), x)


def _mvnormal_mgf(t, mu, cov):
    def _fn(t, mu, cov):
        d = jnp.shape(t)[-1]
        t = t - mu
        cov_inv = jnp.linalg.inv(cov)
        return jnp.exp(
            0.5 * jnp.einsum("...i,...ij,...j->...", t, cov_inv, t)
        ) / jnp.sqrt((2 * jnp.pi) ** d * jnp.linalg.det(cov))

    return jtu.tree_map(lambda tt: _fn(tt, mu, cov), t)


def _mvnormal_cf(t, mu, cov):
    def _fn(t, mu, cov):
        d = jnp.shape(t)[-1]
        t = t - mu
        cov_inv = jnp.linalg.inv(cov)
        return jnp.exp(
            0.5j * jnp.einsum("...i,...ij,...j->...", t, cov_inv, t)
        ) / jnp.sqrt((2 * jnp.pi) ** d * jnp.linalg.det(cov))

    return jtu.tree_map(lambda tt: _fn(tt, mu, cov), t)
