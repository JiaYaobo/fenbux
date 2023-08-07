import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jax.scipy.special import erf, erfinv

from ...core import (
    _check_params_equal_tree_strcutre,
    _intialize_params_tree,
    cdf,
    DTypeLikeFloat,
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
from ...tree_utils import zeros_pytree
from .._base import ContinuousUnivariateDistribution


class LogNormal(ContinuousUnivariateDistribution):
    """LogNormal distribution.
        X ~ LogNormal(μ, σ)

    Args:
        mean (ArrayLike): Mean of the distribution.
        sd (ArrayLike): Standard deviation of the distribution.
        dtype (jax.numpy.dtype): dtype of the distribution, default jnp.float_.
        use_batch (bool): Whether to use with vmap. Default False.
    """

    mean: PyTreeVar
    sd: PyTreeVar

    def __init__(self, mean=0.0, sd=1.0, dtype=jnp.float_, use_batch=False):
        _check_params_equal_tree_strcutre(mean, sd, use_batch=use_batch)
        self.mean, self.sd = _intialize_params_tree(
            mean, sd, use_batch=use_batch, dtype=dtype
        )


@params.dispatch
def _params(d: LogNormal):
    return (d.mean, d.sd)


@support.dispatch
def _support(d: LogNormal):
    return jnp.array([0.0, jnp.inf])


@mean.dispatch
def _mean(d: LogNormal):
    d = d.broadcast_params()
    return jtu.tree_map(lambda m, sd: jnp.exp(m + 0.5 * sd**2), d.mean, d.sd)


@variance.dispatch
def _variance(d: LogNormal):
    d = d.broadcast_params()
    return jtu.tree_map(
        lambda m, sd: (jnp.exp(sd**2) - 1) * jnp.exp(2 * m + sd**2), d.mean, d.sd
    )


@standard_dev.dispatch
def _standard_dev(d: LogNormal):
    d = d.broadcast_params()
    return jtu.tree_map(
        lambda m, sd: jnp.sqrt((jnp.exp(sd**2) - 1) * jnp.exp(2 * m + sd**2)),
        d.mean,
        d.sd,
    )


@skewness.dispatch
def _skewness(d: LogNormal):
    d = d.broadcast_params()
    return jtu.tree_map(
        lambda sd: (jnp.exp(sd**2) + 2) * jnp.sqrt(jnp.exp(sd**2) - 1), d.sd
    )


@kurtosis.dispatch
def _kurtosis(d: LogNormal):
    d = d.broadcast_params()
    return jtu.tree_map(
        lambda sd: jnp.exp(4 * sd**2)
        + 2 * jnp.exp(3 * sd**2)
        + 3 * jnp.exp(2 * sd**2)
        - 6,
        d.sd,
    )


@entropy.dispatch
def _entropy(d: LogNormal):
    d = d.broadcast_params()
    # Here consistent with scipy, but against wikipedia :(
    return jtu.tree_map(
        lambda m, sd: jnp.log(sd * jnp.exp(m + 0.5) * jnp.sqrt(2 * jnp.pi)),
        d.mean,
        d.sd,
    )


@logpdf.dispatch
def _logpdf(d: LogNormal, x):
    d = d.broadcast_params()
    return jtu.tree_map(lambda m, sd: _lognormal_logpdf(x, m, sd), d.mean, d.sd)


@pdf.dispatch
def _pdf(d: LogNormal, x):
    d = d.broadcast_params()
    return jtu.tree_map(lambda m, sd: _lognormal_pdf(x, m, sd), d.mean, d.sd)


@cdf.dispatch
def cdf(d: LogNormal, x):
    d = d.broadcast_params()
    return jtu.tree_map(lambda m, sd: _lognormal_cdf(x, m, sd), d.mean, d.sd)


@logcdf.dispatch
def _logcdf(d: LogNormal, x):
    d = d.broadcast_params()
    return jtu.tree_map(lambda m, sd: _lognormal_logcdf(x, m, sd), d.mean, d.sd)


@quantile.dispatch
def _quantile(d: LogNormal, x):
    d = d.broadcast_params()
    return jtu.tree_map(lambda m, sd: _lognormal_quantile(x, m, sd), d.mean, d.sd)


@sf.dispatch
def _sf(d: LogNormal, x):
    d = d.broadcast_params()
    return jtu.tree_map(lambda m, sd: _lognormal_sf(x, m, sd), d.mean, d.sd)


# Reference: https://en.wikipedia.org/wiki/Log-normal_distribution


def _lognormal_logpdf(x, m, sd):
    def _fn(x, m, sd):
        return (
            -0.5 * jnp.log(2 * jnp.pi)
            - jnp.log(sd)
            - jnp.log(x)
            - ((jnp.log(x) - m) / sd) ** 2 / 2
        )

    return jtu.tree_map(lambda xx: _fn(xx, m, sd), x)


def _lognormal_pdf(x, m, sd):
    def _fn(x, m, sd):
        return jnp.exp(
            -0.5 * jnp.log(2 * jnp.pi)
            - jnp.log(sd)
            - jnp.log(x)
            - ((jnp.log(x) - m) / sd) ** 2 / 2
        )

    return jtu.tree_map(lambda xx: _fn(xx, m, sd), x)


def _lognormal_cdf(x, m, sd):
    def _fn(x, m, sd):
        return 0.5 * (1 + erf((jnp.log(x) - m) / (sd * jnp.sqrt(2))))

    return jtu.tree_map(lambda xx: _fn(xx, m, sd), x)


def _lognormal_logcdf(x, m, sd):
    def _fn(x, m, sd):
        return jnp.log(0.5 * (1 + erf((jnp.log(x) - m) / (sd * jnp.sqrt(2)))))

    return jtu.tree_map(lambda xx: _fn(xx, m, sd), x)


def _lognormal_quantile(x, m, sd):
    def _fn(x, m, sd):
        return jnp.exp(m + sd * jnp.sqrt(2) * erfinv(2 * x - 1))

    return jtu.tree_map(lambda xx: _fn(xx, m, sd), x)


def _lognormal_sf(x, m, sd):
    def _fn(x, m, sd):
        return 1 - _lognormal_cdf(x, m, sd)

    return jtu.tree_map(lambda xx: _fn(xx, m, sd), x)
