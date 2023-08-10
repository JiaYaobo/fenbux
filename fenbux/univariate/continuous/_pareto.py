import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

from ...core import (
    _check_params_equal_tree_strcutre,
    _intialize_params_tree,
    cdf,
    entropy,
    KeyArray,
    kurtosis,
    logcdf,
    logpdf,
    mean,
    params,
    pdf,
    PyTreeVar,
    quantile,
    rand,
    sf,
    Shape,
    skewness,
    standard_dev,
    support,
    variance,
)
from ...dist_special.pareto import (
    pareto_cdf,
    pareto_logcdf,
    pareto_logpdf,
    pareto_pdf,
    pareto_ppf,
    pareto_sf,
)
from ...random_utils import split_tree
from .._base import ContinuousUnivariateDistribution


class Pareto(ContinuousUnivariateDistribution):
    """Pareto distribution.

        X ~ Pareto(shape, scale)

    Args:
        shape (PyTree): Shape parameter of the distribution.
        scale (PyTree): Scale parameter of the distribution.
        dtype (jax.numpy.dtype): dtype of the distribution, default jnp.float_.
        use_batch (bool): Whether to use with vmap. Default False.

    Examples:
        >>> import jax.numpy as jnp
        >>> from fenbux import Pareto, logpdf
        >>> dist = Pareto(1.0, 1.0)
        >>> logpdf(dist, jnp.ones((10, )))
    """

    shape: PyTreeVar
    scale: PyTreeVar

    def __init__(self, shape=0.0, scale=0.0, dtype=jnp.float_, use_batch=False):
        _check_params_equal_tree_strcutre(shape, scale, use_batch=use_batch)
        self.shape, self.scale = _intialize_params_tree(
            shape, scale, use_batch=use_batch, dtype=dtype
        )


@params.dispatch
def _params(d: Pareto):
    return (d.shape, d.scale)


@support.dispatch
def _support(d: Pareto):
    d = d.broadcast_params()
    return jtu.tree_map(lambda _, scale: (scale, jnp.inf), d.shape, d.scale)


@mean.dispatch
def _mean(d: Pareto):
    d = d.broadcast_params()
    return jtu.tree_map(
        lambda a, b: jnp.where(a > 1.0, b * a / (a - 1.0), jnp.inf), d.shape, d.scale
    )


@variance.dispatch
def _variance(d: Pareto):
    d = d.broadcast_params()
    return jtu.tree_map(
        lambda a, b: jnp.where(
            a > 2.0, b**2 * a / ((a - 1.0) ** 2 * (a - 2.0)), jnp.inf
        ),
        d.shape,
        d.scale,
    )


@standard_dev.dispatch
def _standard_dev(d: Pareto):
    d = d.broadcast_params()
    return jtu.tree_map(
        lambda a, b: jnp.where(
            a > 2.0, b * jnp.sqrt(a) / ((a - 1.0) * jnp.sqrt(a - 2.0)), jnp.inf
        ),
        d.shape,
        d.scale,
    )


@skewness.dispatch
def _skewness(d: Pareto):
    d = d.broadcast_params()
    return jtu.tree_map(
        lambda a: jnp.where(
            a > 3, 2.0 * (1.0 + a) / (a - 3.0) * jnp.sqrt((a - 2.0) / a), jnp.nan
        ),
        d.shape,
    )


@kurtosis.dispatch
def _kurtosis(d: Pareto):
    d = d.broadcast_params()
    return jtu.tree_map(
        lambda a: jnp.where(
            a > 4,
            6.0 * (a**3 + a**2 - 6.0 * a - 2.0) / (a * (a - 3.0) * (a - 4.0)),
            jnp.nan,
        ),
        d.shape,
    )


@entropy.dispatch
def _entropy(d: Pareto):
    d = d.broadcast_params()
    return jtu.tree_map(
        lambda a, b: jnp.log(b) - jnp.log(a) + 1.0 / a + 1.0, d.shape, d.scale
    )


@logpdf.dispatch
def _logpdf(d: Pareto, x):
    d = d.broadcast_params()
    return jtu.tree_map(lambda a, b: _pareto_logpdf(x, a, b), d.shape, d.scale)


@pdf.dispatch
def _pdf(d: Pareto, x):
    d = d.broadcast_params()
    return jtu.tree_map(lambda a, b: _pareto_pdf(x, a, b), d.shape, d.scale)


@logcdf.dispatch
def _logcdf(d: Pareto, x):
    d = d.broadcast_params()
    return jtu.tree_map(lambda a, b: _pareto_logcdf(x, a, b), d.shape, d.scale)


@cdf.dispatch
def _cdf(d: Pareto, x):
    d = d.broadcast_params()
    return jtu.tree_map(lambda a, b: _pareto_cdf(x, a, b), d.shape, d.scale)


@quantile.dispatch
def _quantile(d: Pareto, x):
    d = d.broadcast_params()
    return jtu.tree_map(lambda a, b: _pareto_quantile(x, a, b), d.shape, d.scale)


@sf.dispatch
def _sf(d: Pareto, x):
    d = d.broadcast_params()
    return jtu.tree_map(lambda a, b: _pareto_sf(x, a, b), d.shape, d.scale)


@rand.dispatch
def _rand(d: Pareto, key: KeyArray, shape: Shape = (), dtype: jnp.dtype = jnp.float_):
    d = d.broadcast_params()
    _key_tree = split_tree(key, d.shape)
    return jtu.tree_map(
        lambda a, b, k: jr.pareto(k, a, shape, dtype) * b, d.shape, d.scale, _key_tree
    )


def _pareto_logpdf(x, shape, scale):
    return jtu.tree_map(lambda xx: pareto_logpdf(xx, shape, scale), x)


def _pareto_pdf(x, shape, scale):
    return jtu.tree_map(lambda xx: pareto_pdf(xx, shape, scale), x)


def _pareto_logcdf(x, shape, scale):
    return jtu.tree_map(lambda xx: pareto_logcdf(xx, shape, scale), x)


def _pareto_cdf(x, shape, scale):
    return jtu.tree_map(lambda xx: pareto_cdf(xx, shape, scale), x)


def _pareto_quantile(x, shape, scale):
    return jtu.tree_map(lambda xx: pareto_ppf(xx, shape, scale), x)


def _pareto_sf(x, shape, scale):
    return jtu.tree_map(lambda xx: pareto_sf(xx, shape, scale), x)
