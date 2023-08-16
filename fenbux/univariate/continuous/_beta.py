import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

from ...core import (
    _cdf_impl,
    _check_params_equal_tree_strcutre,
    _intialize_params_tree,
    _logcdf_impl,
    _logpdf_impl,
    _pdf_impl,
    _quantile_impl,
    _sf_impl,
    DTypeLikeFloat,
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
from ...dist_special.beta import (
    beta_cdf,
    beta_logcdf,
    beta_logpdf,
    beta_pdf,
    beta_ppf,
    beta_sf,
)
from ...random_utils import split_tree
from .._base import ContinuousUnivariateDistribution


class Beta(ContinuousUnivariateDistribution):
    """Beta distribution.

    Args:
        a (ArrayLike): Shape parameter a.
        b (ArrayLike): Shape parameter b.
        dtype (jax.numpy.dtype): dtype of the distribution, default jnp.float_.
        use_batch (bool): Whether to use with vmap. Default False.

    Examples:
        >>> import jax.numpy as jnp
        >>> from fenbux import Beta, logpdf
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


@params.dispatch
def params(dist: Beta):
    return dist.a, dist.b


@support.dispatch
def _domain(d: Beta):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda _: (0.0, jnp.inf), _tree.a)


@mean.dispatch
def _mean(d: Beta):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda a, b: a / (a + b), _tree.a, _tree.b)


@variance.dispatch
def _variance(d: Beta):
    _tree = d.broadcast_params()
    return jtu.tree_map(
        lambda a, b: a * b / ((a + b) ** 2 * (a + b + 1)), _tree.a, _tree.b
    )


@standard_dev.dispatch
def _standard_dev(d: Beta):
    _tree = d.broadcast_params()
    return jtu.tree_map(
        lambda a, b: jnp.sqrt(a * b / ((a + b) ** 2 * (a + b + 1))), _tree.a, _tree.b
    )


@skewness.dispatch
def _skewness(d: Beta):
    _tree = d.broadcast_params()
    return jtu.tree_map(
        lambda a, b: 2
        * (b - a)
        * jnp.sqrt(a + b + 1)
        / ((a + b + 2) * jnp.sqrt(a * b)),
        _tree.a,
        _tree.b,
    )


@kurtosis.dispatch
def _kurtosis(d: Beta):
    _tree = d.broadcast_params()
    return jtu.tree_map(
        lambda a, b: (
            6
            * ((a - b) ** 2 * (a + b + 1) - a * b * (a + b + 2))
            / (a * b * (a + b + 2) * (a + b + 3))
        ),
        _tree.a,
        _tree.b,
    )


@_logpdf_impl.dispatch
def _logpdf(d: Beta, x: PyTreeVar):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda a, b: _beta_log_pdf(x, a, b), _tree.a, _tree.b)


@_pdf_impl.dispatch
def _pdf(d: Beta, x: PyTreeVar):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda a, b: _beta_pdf(x, a, b), _tree.a, _tree.b)


@_logcdf_impl.dispatch
def _logcdf(d: Beta, x: PyTreeVar):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda a, b: _beta_log_cdf(x, a, b), _tree.a, _tree.b)


@_cdf_impl.dispatch
def _cdf(d: Beta, x: PyTreeVar):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda a, b: _beta_cdf(x, a, b), _tree.a, _tree.b)


@_quantile_impl.dispatch
def _quantile(d: Beta, x: PyTreeVar):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda a, b: _beta_quantile(x, a, b), _tree.a, _tree.b)


@_sf_impl.dispatch
def _sf(d: Beta, x: PyTreeVar):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda a, b: _beta_sf(x, a, b), _tree.a, _tree.b)


@rand.dispatch
def _rand(
    d: Beta, key: KeyArray, shape: Shape = (), dtype: DTypeLikeFloat = jnp.float_
):
    _tree = d.broadcast_params()
    _key_tree = split_tree(key, _tree.a)
    return jtu.tree_map(
        lambda a, b, k: jr.beta(k, a, b, shape=shape, dtype=dtype),
        _tree.a,
        _tree.b,
        _key_tree,
    )


def _beta_log_pdf(x, a, b):
    return jtu.tree_map(lambda xx: beta_logpdf(xx, a, b), x)


def _beta_pdf(x, a, b):
    return jtu.tree_map(lambda xx: beta_pdf(xx, a, b), x)


def _beta_log_cdf(x, a, b):
    return jtu.tree_map(lambda xx: beta_logcdf(xx, a, b), x)


def _beta_cdf(x, a, b):
    return jtu.tree_map(lambda xx: beta_cdf(xx, a, b), x)


def _beta_quantile(x, a, b):
    return jtu.tree_map(lambda xx: beta_ppf(xx, a, b), x)


def _beta_sf(x, a, b):
    return jtu.tree_map(lambda xx: beta_sf(xx, a, b), x)
