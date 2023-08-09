import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jax.scipy.special import gamma

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
from ...random_utils import split_tree
from .._base import ContinuousUnivariateDistribution


class Weibull(ContinuousUnivariateDistribution):
    """Weibull distribution.

        X ~ Weibull(shape, scale)

    Args:
        shape (PyTree): Shape parameter of the distribution.
        scale (PyTree): Scale parameter of the distribution.
        dtype (jax.numpy.dtype): dtype of the distribution, default jnp.float_.
        use_batch (bool): Whether to use with vmap. Default False.

    Examples:
        >>> import jax.numpy as jnp
        >>> from fenbux import WeiBull, logpdf
        >>> dist = WeiBull(1.0, 1.0)
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
def _params(d: Weibull):
    return (d.shape, d.scale)


@support.dispatch
def _support(d: Weibull):
    d = d.broadcast_params()
    return jtu.tree_map(lambda _: (0.0, jnp.inf), d.shape, d.scale)


@mean.dispatch
def _mean(d: Weibull):
    d = d.broadcast_params()
    return jtu.tree_map(
        lambda shape, scale: scale * gamma(1.0 + 1.0 / shape), d.shape, d.scale
    )


@variance.dispatch
def _variance(d: Weibull):
    d = d.broadcast_params()
    return jtu.tree_map(
        lambda shape, scale: scale**2 * gamma(1.0 + 2.0 / shape) - mean(d) ** 2,
        d.shape,
        d.scale,
    )


@standard_dev.dispatch
def _standard_dev(d: Weibull):
    d = d.broadcast_params()
    return jtu.tree_map(
        lambda shape, scale: scale * jnp.sqrt(gamma(1.0 + 2.0 / shape) - mean(d) ** 2),
        d.shape,
        d.scale,
    )


@skewness.dispatch
def _skewness(d: Weibull):
    d = d.broadcast_params()

    def _fn(shape):
        gamma1 = gamma(1.0 + 1.0 / shape)
        gamma2 = gamma(1.0 + 2.0 / shape)
        gamma3 = gamma(1.0 + 3.0 / shape)
        return (gamma3 - 3 * gamma2 * gamma1 + 2 * gamma1**3) / (
            gamma2 - gamma1**2
        ) ** 1.5

    return jtu.tree_map(
        lambda shape: _fn(shape),
        d.shape,
    )


@kurtosis.dispatch
def _kurtosis(d: Weibull):
    d = d.broadcast_params()

    def _fn(shape):
        gamma1 = gamma(1.0 + 1.0 / shape)
        gamma2 = gamma(1.0 + 2.0 / shape)
        gamma3 = gamma(1.0 + 3.0 / shape)
        gamma4 = gamma(1.0 + 4.0 / shape)
        return (
            gamma4 - 4 * gamma3 * gamma1 + 6 * gamma2 * gamma1**2 - 3 * gamma1**4
        ) / (gamma2 - gamma1**2) ** 2

    return jtu.tree_map(
        lambda shape: _fn(shape),
        d.shape,
    )


@entropy.dispatch
def _entropy(d: Weibull):
    d = d.broadcast_params()
    return jtu.tree_map(
        lambda shape, scale: (
            jnp.euler_gamma * (1.0 - 1.0 / shape) + jnp.log(scale / shape) + 1.0
        ),
        d.shape,
        d.scale,
    )


@logpdf.dispatch
def _logpdf(d: Weibull, x: PyTreeVar):
    d = d.broadcast_params()
    return jtu.tree_map(
        lambda shape, scale, xx: _weibull_logpdf(xx, shape, scale), d.shape, d.scale, x
    )


@pdf.dispatch
def _pdf(d: Weibull, x: PyTreeVar):
    d = d.broadcast_params()
    return jtu.tree_map(
        lambda shape, scale, xx: _weibull_pdf(xx, shape, scale), d.shape, d.scale, x
    )


@cdf.dispatch
def _cdf(d: Weibull, x: PyTreeVar):
    d = d.broadcast_params()
    return jtu.tree_map(
        lambda shape, scale, xx: _weibull_cdf(xx, shape, scale), d.shape, d.scale, x
    )


@logcdf.dispatch
def _logcdf(d: Weibull, x: PyTreeVar):
    d = d.broadcast_params()
    return jtu.tree_map(
        lambda shape, scale, xx: _weibull_logcdf(xx, shape, scale), d.shape, d.scale, x
    )


@quantile.dispatch
def _quantile(d: Weibull, x: PyTreeVar):
    d = d.broadcast_params()
    return jtu.tree_map(
        lambda shape, scale, xx: _weibull_quantile(xx, shape, scale),
        d.shape,
        d.scale,
        x,
    )


@sf.dispatch
def _sf(d: Weibull, x: PyTreeVar):
    d = d.broadcast_params()
    return jtu.tree_map(
        lambda shape, scale, xx: _weibull_sf(xx, shape, scale), d.shape, d.scale, x
    )


@rand.dispatch
def _rand(d: Weibull, key: KeyArray, shape: Shape = (), dtype: jnp.dtype = jnp.float_):
    d = d.broadcast_params()
    _key_tree = split_tree(key, d.shape)
    return jtu.tree_map(
        lambda shape_, scale, key: jr.weibull_min(
            key, scale=scale, concentration=shape_, shape=shape, dtype=dtype
        ),
        d.shape,
        d.scale,
        _key_tree,
    )


def _weibull_logpdf(x, shape, scale):
    def _fn(x, shape, scale):
        return jnp.where(
            x < 0,
            0,
            jnp.log(shape / scale)
            + (shape - 1) * jnp.log(x / scale)
            - (x / scale) ** shape,
        )

    return jtu.tree_map(lambda xx: _fn(xx, shape, scale), x)


def _weibull_pdf(x, shape, scale):
    def _fn(x, shape, scale):
        return jnp.where(
            x < 0,
            0,
            (shape / scale)
            * (x / scale) ** (shape - 1)
            * jnp.exp(-((x / scale) ** shape)),
        )

    return jtu.tree_map(lambda xx: _fn(xx, shape, scale), x)


def _weibull_cdf(x, shape, scale):
    def _fn(x, shape, scale):
        return jnp.where(
            x < 0,
            0,
            1 - jnp.exp(-((x / scale) ** shape)),
        )

    return jtu.tree_map(lambda xx: _fn(xx, shape, scale), x)


def _weibull_logcdf(x, shape, scale):
    def _fn(x, shape, scale):
        return jnp.where(
            x < 0,
            0,
            jnp.log(1 - jnp.exp(-((x / scale) ** shape))),
        )

    return jtu.tree_map(lambda xx: _fn(xx, shape, scale), x)


def _weibull_quantile(x, shape, scale):
    def _fn(x, shape, scale):
        return scale * (-jnp.log(1 - x)) ** (1 / shape)

    return jtu.tree_map(lambda xx: _fn(xx, shape, scale), x)


def _weibull_sf(x, shape, scale):
    def _fn(x, shape, scale):
        return jnp.where(
            x < 0,
            1,
            jnp.exp(-((x / scale) ** shape)),
        )

    return jtu.tree_map(lambda xx: _fn(xx, shape, scale), x)
