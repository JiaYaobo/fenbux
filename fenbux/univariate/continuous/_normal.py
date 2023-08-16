import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

from ...core import (
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
    entropy,
    KeyArray,
    kurtosis,
    mean,
    params,
    PyTreeVar,
    rand,
    Shape,
    skewness,
    standard_dev,
    support,
    variance,
)
from ...dist_special.normal import (
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
from ...tree_utils import zeros_pytree
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
        >>> from fenbux import Normal, logpdf
        >>> dist = Normal(0.0, jnp.ones((10, )))
        >>> vmap(logpdf, in_axes=(Normal(None, 0, use_batch=True), 0))(dist, jnp.zeros((10, )))
        Array([-0.9189385, -0.9189385, -0.9189385, -0.9189385, -0.9189385,
            -0.9189385, -0.9189385, -0.9189385, -0.9189385, -0.9189385],      dtype=float32)


    Attributes:
        mean (PyTree): Mean of the distribution.
        sd (PyTree): Standard deviation of the distribution.
    """

    mean: PyTreeVar
    sd: PyTreeVar

    def __init__(self, mean=0.0, sd=0.0, dtype=jnp.float_, use_batch=False):
        _check_params_equal_tree_strcutre(mean, sd, use_batch=use_batch)
        self.mean, self.sd = _intialize_params_tree(
            mean, sd, use_batch=use_batch, dtype=dtype
        )


@params.dispatch
def _params(d: Normal):
    return (d.mean, d.sd)


@support.dispatch
def _domain(d: Normal):
    _tree = d.broadcast_params().mean
    return jtu.tree_map(lambda _: (-jnp.inf, jnp.inf), _tree)


@mean.dispatch
def _mean(d: Normal):
    return d.broadcast_params().mean


@variance.dispatch
def _variance(d: Normal):
    return jtu.tree_map(lambda x: x**2, d.broadcast_params().sd)


@standard_dev.dispatch
def _std(d: Normal):
    return d.broadcast_params().sd


@kurtosis.dispatch
def _kurtosis(d: Normal):
    shape = d.broadcast_shapes()
    return zeros_pytree(shape)


@skewness.dispatch
def _skewness(d: Normal):
    shape = d.broadcast_shapes()
    return zeros_pytree(shape)


@entropy.dispatch
def _entropy(d: Normal):
    d = d.broadcast_params()
    entropy = jtu.tree_map(lambda σ: 0.5 * jnp.log(2 * jnp.pi * σ**2) + 0.5, d.sd)
    return entropy


@_logpdf_impl.dispatch
def _logpdf(d: Normal, x: PyTreeVar):
    d = d.broadcast_params()
    log_d = jtu.tree_map(lambda μ, σ: _normal_log_pdf(x, μ, σ), d.mean, d.sd)
    return log_d


@_pdf_impl.dispatch
def _pdf(d: Normal, x: PyTreeVar):
    d = d.broadcast_params()
    d = jtu.tree_map(lambda μ, σ: _normal_pdf(x, μ, σ), d.mean, d.sd)
    return d


@_logcdf_impl.dispatch
def _logcdf(d: Normal, x: PyTreeVar):
    d = d.broadcast_params()
    log_cdf = jtu.tree_map(lambda μ, σ: _normal_log_cdf(x, μ, σ), d.mean, d.sd)
    return log_cdf


@_cdf_impl.dispatch
def _cdf(d: Normal, x: PyTreeVar):
    d = d.broadcast_params()
    prob = jtu.tree_map(lambda μ, σ: _normal_cdf(x, μ, σ), d.mean, d.sd)
    return prob


@_quantile_impl.dispatch
def _quantile(d: Normal, q: PyTreeVar):
    d = d.broadcast_params()
    x = jtu.tree_map(lambda μ, σ: _normal_quantile(q, μ, σ), d.mean, d.sd)
    return x


@rand.dispatch
def _rand(
    d: Normal, key: KeyArray, shape: Shape = (), dtype: DTypeLikeFloat = jnp.float_
):
    d = d.broadcast_params()
    _key_tree = split_tree(key, d.mean)
    rvs = jtu.tree_map(
        lambda μ, σ, key: jr.normal(key, shape, dtype=dtype) * σ + μ,
        d.mean,
        d.sd,
        _key_tree,
    )
    return rvs


@_mgf_impl.dispatch
def _mgf(d: Normal, t: PyTreeVar):
    d = d.broadcast_params()
    mgf = jtu.tree_map(lambda μ, σ: _normal_mgf(t, μ, σ), d.mean, d.sd)
    return mgf


@_cf_impl.dispatch
def _cf(d: Normal, t: PyTreeVar):
    d = d.broadcast_params()
    cf = jtu.tree_map(lambda μ, σ: _normal_cf(t, μ, σ), d.mean, d.sd)
    return cf


@_sf_impl.dispatch
def _sf(d: Normal, x: PyTreeVar):
    d = d.broadcast_params()
    sf = jtu.tree_map(lambda μ, σ: _normal_sf(x, μ, σ), d.mean, d.sd)
    return sf


def _normal_cf(t, μ, σ):
    return jtu.tree_map(lambda tt: normal_cf(tt, μ, σ), t)


def _normal_mgf(t, μ, σ):
    return jtu.tree_map(lambda tt: normal_mgf(tt, μ, σ), t)


def _normal_pdf(x, μ, σ):
    return jtu.tree_map(lambda xx: normal_pdf(xx, μ, σ), x)


def _normal_log_pdf(x, μ, σ):
    return jtu.tree_map(lambda xx: normal_logpdf(xx, μ, σ), x)


def _normal_log_cdf(x, μ, σ):
    return jtu.tree_map(lambda xx: normal_logcdf(xx, μ, σ), x)


def _normal_cdf(x, μ, σ):
    return jtu.tree_map(lambda xx: normal_cdf(xx, μ, σ), x)


def _normal_quantile(x, μ, σ):
    return jtu.tree_map(lambda xx: normal_ppf(xx, μ, σ), x)


def _normal_sf(x, μ, σ):
    return jtu.tree_map(lambda xx: normal_sf(xx, μ, σ), x)
