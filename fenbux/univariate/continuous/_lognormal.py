import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
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
from ...dist_math.lognormal import (
    lognormal_cdf,
    lognormal_logcdf,
    lognormal_logpdf,
    lognormal_pdf,
    lognormal_ppf,
    lognormal_sf,
)
from ...random_utils import split_tree
from ...tree_utils import tree_map_dist_at
from .._base import ContinuousUnivariateDistribution


class LogNormal(ContinuousUnivariateDistribution):
    """LogNormal distribution.
        X ~ LogNormal(μ, σ)

    Args:
        mean (ArrayLike): Mean of the distribution.
        sd (ArrayLike): Standard deviation of the distribution.
        dtype (jax.numpy.dtype): dtype of the distribution, default jnp.float_.
        use_batch (bool): Whether to use with vmap. Default False.
        
    Examples:
        >>> import jax.numpy as jnp
        >>> from fenbux import logpdf
        >>> from fenbux.univariate import LogNormal
        >>> dist = LogNormal(1.0, 1.0)
        >>> logpdf(dist, jnp.ones((10, )))
    """

    mean: PyTreeVar
    sd: PyTreeVar

    def __init__(self, mean=0.0, sd=1.0, dtype=jnp.float_, use_batch=False):
        _check_params_equal_tree_strcutre(mean, sd, use_batch=use_batch)
        self.mean, self.sd = _intialize_params_tree(
            mean, sd, use_batch=use_batch, dtype=dtype
        )


@_params_impl.dispatch
def _params(d: LogNormal):
    return (d.mean, d.sd)


@_support_impl.dispatch
def _support(d: LogNormal):
    d = d.broadcast_params()
    return jtu.tree_map(lambda m: jnp.zeros_like(m), d.mean), jtu.tree_map(
        lambda m: jnp.full_like(m, jnp.inf), d.mean
    )


@_mean_impl.dispatch
def _mean(d: LogNormal):
    d = d.broadcast_params()
    return jtu.tree_map(lambda m, sd: jnp.exp(m + 0.5 * sd**2), d.mean, d.sd)


@_variance_impl.dispatch
def _variance(d: LogNormal):
    d = d.broadcast_params()
    return jtu.tree_map(
        lambda m, sd: (jnp.exp(sd**2) - 1) * jnp.exp(2 * m + sd**2), d.mean, d.sd
    )


@_standard_dev_impl.dispatch
def _standard_dev(d: LogNormal):
    d = d.broadcast_params()
    return jtu.tree_map(
        lambda m, sd: jnp.sqrt((jnp.exp(sd**2) - 1) * jnp.exp(2 * m + sd**2)),
        d.mean,
        d.sd,
    )


@_skewness_impl.dispatch
def _skewness(d: LogNormal):
    d = d.broadcast_params()
    return jtu.tree_map(
        lambda sd: (jnp.exp(sd**2) + 2) * jnp.sqrt(jnp.exp(sd**2) - 1), d.sd
    )


@_kurtosis_impl.dispatch
def _kurtosis(d: LogNormal):
    d = d.broadcast_params()
    return jtu.tree_map(
        lambda sd: jnp.exp(4 * sd**2)
        + 2 * jnp.exp(3 * sd**2)
        + 3 * jnp.exp(2 * sd**2)
        - 6,
        d.sd,
    )


@_entropy_impl.dispatch
def _entropy(d: LogNormal):
    d = d.broadcast_params()
    # Here consistent with scipy, but against wikipedia :(
    return jtu.tree_map(
        lambda m, sd: jnp.log(sd * jnp.exp(m + 0.5) * jnp.sqrt(2 * jnp.pi)),
        d.mean,
        d.sd,
    )


@_logpdf_impl.dispatch
def _logpdf(d: LogNormal, x: ArrayLike):
    d = d.broadcast_params()
    return tree_map_dist_at(lognormal_logpdf, d, x)


@_pdf_impl.dispatch
def _pdf(d: LogNormal, x: ArrayLike):
    d = d.broadcast_params()
    return tree_map_dist_at(lognormal_pdf, d, x)


@_cdf_impl.dispatch
def _cdf_impl(d: LogNormal, x: ArrayLike):
    d = d.broadcast_params()
    return tree_map_dist_at(lognormal_cdf, d, x)


@_logcdf_impl.dispatch
def _logcdf(d: LogNormal, x: ArrayLike):
    d = d.broadcast_params()
    return tree_map_dist_at(lognormal_logcdf, d, x)


@_quantile_impl.dispatch
def _quantile(d: LogNormal, x: ArrayLike):
    d = d.broadcast_params()
    return tree_map_dist_at(lognormal_ppf, d, x)


@_sf_impl.dispatch
def _sf(d: LogNormal, x: ArrayLike):
    d = d.broadcast_params()
    return tree_map_dist_at(lognormal_sf, d, x)


@_rand_impl.dispatch
def _rand(
    d: LogNormal, key: KeyArray, shape: Shape = (), dtype: DTypeLikeFloat = float
):
    d = d.broadcast_params()
    _key_tree = split_tree(key, d.mean)
    return jtu.tree_map(
        lambda m, sd, k: jr.lognormal(k, sd, shape=shape, dtype=dtype) * jnp.exp(m),
        d.mean,
        d.sd,
        _key_tree,
    )
