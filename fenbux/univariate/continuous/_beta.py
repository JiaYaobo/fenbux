import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import ArrayLike

from ...core import (
    _cdf_impl,
    _check_params_equal_tree_strcutre,
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
from ...dist_math.beta import (
    beta_cdf,
    beta_logcdf,
    beta_logpdf,
    beta_pdf,
    beta_ppf,
    beta_sf,
)
from ...random_utils import split_tree
from ...tree_utils import tree_map_dist_at
from .._base import ContinuousUnivariateDistribution


class Beta(ContinuousUnivariateDistribution):
    """Beta distribution.

    Args:
        a (PyTreeVar): Shape parameter a.
        b (PyTreeVar): Shape parameter b.
        dtype (jax.numpy.dtype): dtype of the distribution, default jnp.float_.
        use_batch (bool): Whether to use with vmap. Default False.

    Examples:
        >>> import jax.numpy as jnp
        >>> from fenbux import logpdf
        >>> from fenbux.univariate import Beta
        >>> dist = Beta(1.0, 1.0)
        >>> logpdf(dist, jnp.ones((10, )))
    """

    a: PyTreeVar
    b: PyTreeVar

    def __init__(
        self, a: PyTreeVar = 0.0, b: PyTreeVar = 0.0, dtype=jnp.float_, use_batch=False
    ):
        _check_params_equal_tree_strcutre(a, b, use_batch=use_batch)
        self.a, self.b = _intialize_params_tree(a, b, use_batch=use_batch, dtype=dtype)


@_params_impl.dispatch
def _params_impl(dist: Beta):
    return dist.a, dist.b


@_support_impl.dispatch
def _support(d: Beta):
    dist = d.broadcast_params()
    return jtu.tree_map(lambda a: jnp.zeros_like(a), dist.a), jtu.tree_map(
        lambda a: jnp.ones_like(a), dist.a
    )


@_mean_impl.dispatch
def _mean(d: Beta):
    dist = d.broadcast_params()
    return jtu.tree_map(lambda a, b: a / (a + b), dist.a, dist.b)


@_variance_impl.dispatch
def _variance(d: Beta):
    dist = d.broadcast_params()
    return jtu.tree_map(
        lambda a, b: a * b / ((a + b) ** 2 * (a + b + 1)), dist.a, dist.b
    )


@_standard_dev_impl.dispatch
def _standard_dev(d: Beta):
    dist = d.broadcast_params()
    return jtu.tree_map(
        lambda a, b: jnp.sqrt(a * b / ((a + b) ** 2 * (a + b + 1))), dist.a, dist.b
    )


@_skewness_impl.dispatch
def _skewness(d: Beta):
    dist = d.broadcast_params()
    return jtu.tree_map(
        lambda a, b: 2
        * (b - a)
        * jnp.sqrt(a + b + 1)
        / ((a + b + 2) * jnp.sqrt(a * b)),
        dist.a,
        dist.b,
    )


@_kurtosis_impl.dispatch
def _kurtosis(d: Beta):
    dist = d.broadcast_params()
    return jtu.tree_map(
        lambda a, b: (
            6
            * ((a - b) ** 2 * (a + b + 1) - a * b * (a + b + 2))
            / (a * b * (a + b + 2) * (a + b + 3))
        ),
        dist.a,
        dist.b,
    )


@_logpdf_impl.dispatch
def _logpdf(d: Beta, x: ArrayLike):
    dist = d.broadcast_params()
    return tree_map_dist_at(beta_logpdf, dist, x)


@_pdf_impl.dispatch
def _pdf(d: Beta, x: ArrayLike):
    return tree_map_dist_at(beta_pdf, d, x)


@_logcdf_impl.dispatch
def _logcdf(d: Beta, x: ArrayLike):
    dist = d.broadcast_params()
    return tree_map_dist_at(beta_logcdf, dist, x)


@_cdf_impl.dispatch
def _cdf(d: Beta, x: ArrayLike):
    dist = d.broadcast_params()
    return tree_map_dist_at(beta_cdf, dist, x)


@_quantile_impl.dispatch
def _quantile(d: Beta, x: ArrayLike):
    dist = d.broadcast_params()
    return tree_map_dist_at(beta_ppf, dist, x)


@_sf_impl.dispatch
def _sf(d: Beta, x: ArrayLike):
    dist = d.broadcast_params()
    return tree_map_dist_at(beta_sf, dist, x)


@_rand_impl.dispatch
def _rand(d: Beta, key: KeyArray, shape: Shape = (), dtype: DTypeLikeFloat = float):
    dist = d.broadcast_params()
    _key_tree = split_tree(key, dist.a)
    return jtu.tree_map(
        lambda a, b, k: jr.beta(k, a, b, shape=shape, dtype=dtype),
        dist.a,
        dist.b,
        _key_tree,
    )
