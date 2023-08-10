import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

from ...core import (
    _check_params_equal_tree_strcutre,
    _intialize_params_tree,
    cdf,
    cf,
    KeyArray,
    kurtosis,
    logcdf,
    logpdf,
    mean,
    mgf,
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
from ...dist_special.gamma import (
    gamma_cdf,
    gamma_cf,
    gamma_logcdf,
    gamma_logpdf,
    gamma_mgf,
    gamma_pdf,
    gamma_ppf,
    gamma_sf,
)
from ...random_utils import split_tree
from .._base import ContinuousUnivariateDistribution


class Gamma(ContinuousUnivariateDistribution):
    """Gamma distribution.

        X ~ Gamma(shape, rate)

    Args:
        shape (PyTree): Shape parameter of the distribution.
        rate (PyTree): Rate parameter of the distribution.
        dtype (jax.numpy.dtype): dtype of the distribution, default jnp.float_.
        use_batch (bool): Whether to use with vmap. Default False.

    Examples:
        >>> import jax.numpy as jnp
        >>> from fenbux import Gamma, logpdf
        >>> dist = Gamma(1.0, 1.0)
        >>> logpdf(dist, jnp.ones((10, )))
    """

    shape: PyTreeVar
    rate: PyTreeVar

    def __init__(self, shape=0.0, rate=0.0, dtype=jnp.float_, use_batch=False):
        _check_params_equal_tree_strcutre(shape, rate, use_batch=use_batch)
        self.shape, self.rate = _intialize_params_tree(
            shape, rate, use_batch=use_batch, dtype=dtype
        )


@params.dispatch
def _params(d: Gamma):
    return (d.shape, d.rate)


@support.dispatch
def _domain(d: Gamma):
    d = d.broadcast_params().shape
    return jtu.tree_map(lambda _: (0.0, jnp.inf), d)


@mean.dispatch
def _mean(d: Gamma):
    d = d.broadcast_params()
    return jtu.tree_map(lambda α, β: α / β, d.shape, d.rate)


@variance.dispatch
def _variance(d: Gamma):
    d = d.broadcast_params()
    return jtu.tree_map(lambda α, β: α / (β**2), d.shape, d.rate)


@standard_dev.dispatch
def _std(d: Gamma):
    d = d.broadcast_params()
    return jtu.tree_map(lambda α, β: jnp.sqrt(α / (β**2)), d.shape, d.rate)


@kurtosis.dispatch
def _kurtosis(d: Gamma):
    d = d.broadcast_params()
    return jtu.tree_map(lambda α: 6 / α, d.shape)


@skewness.dispatch
def _skewness(d: Gamma):
    d = d.broadcast_params()
    return jtu.tree_map(lambda α: 2 / jnp.sqrt(α), d.shape)


@logpdf.dispatch
def _logpdf(d: Gamma, x: PyTreeVar):
    d = d.broadcast_params()
    log_d = jtu.tree_map(lambda α, β: _gamma_log_pdf(x, α, β), d.shape, d.rate)
    return log_d


@pdf.dispatch
def _pdf(d: Gamma, x: PyTreeVar):
    d = d.broadcast_params()
    return jtu.tree_map(lambda α, β: _gamma_pdf(x, α, β), d.shape, d.rate)


@logcdf.dispatch
def _logcdf(d: Gamma, x: PyTreeVar):
    d = d.broadcast_params()
    log_cdf = jtu.tree_map(lambda α, β: _gamma_log_cdf(x, α, β), d.shape, d.rate)
    return log_cdf


@cdf.dispatch
def _cdf(d: Gamma, x: PyTreeVar):
    d = d.broadcast_params()
    prob = jtu.tree_map(lambda α, β: _gamma_cdf(x, α, β), d.shape, d.rate)
    return prob


@quantile.dispatch
def _quantile(d: Gamma, q: PyTreeVar):
    d = d.broadcast_params()
    x = jtu.tree_map(lambda α, β: _gamma_quantile(q, α, β), d.shape, d.rate)
    return x


@rand.dispatch
def _rand(d: Gamma, key: KeyArray, shape: Shape = (), dtype=jnp.float_):
    d = d.broadcast_params()
    _key_tree = split_tree(key, d.shape)
    rvs = jtu.tree_map(
        lambda α, β, key: jr.gamma(key, shape, dtype=dtype) * β + α,
        d.shape,
        d.rate,
        _key_tree,
    )
    return rvs


@mgf.dispatch
def _mgf(d: Gamma, t: PyTreeVar):
    d = d.broadcast_params()
    mgf = jtu.tree_map(lambda α, β: _gamma_mgf(t, α, β), d.shape, d.rate)
    return mgf


@cf.dispatch
def _cf(d: Gamma, t: PyTreeVar):
    d = d.broadcast_params()
    cf = jtu.tree_map(lambda α, β: _gamma_cf(t, α, β), d.shape, d.rate)
    return cf


@sf.dispatch
def _sf(d: Gamma, x: PyTreeVar):
    d = d.broadcast_params()
    sf = jtu.tree_map(lambda α, β: _gamma_sf(x, α, β), d.shape, d.rate)
    return sf


def _gamma_log_pdf(x, α, β):
    return jtu.tree_map(lambda xx: gamma_logpdf(xx, α, β), x)


def _gamma_pdf(x, α, β):
    return jtu.tree_map(lambda xx: gamma_pdf(xx, α, β), x)


def _gamma_log_cdf(x, α, β):
    return jtu.tree_map(lambda xx: gamma_logcdf(xx, α, β), x)


def _gamma_cdf(x, α, β):
    return jtu.tree_map(lambda xx: gamma_cdf(xx, α, β), x)


def _gamma_quantile(x, α, β):
    return jtu.tree_map(lambda xx: gamma_ppf(xx, α, β), x)


def _gamma_mgf(t, α, β):
    return jtu.tree_map(lambda tt: gamma_mgf(tt, α, β), t)


def _gamma_cf(t, α, β):
    return jtu.tree_map(lambda tt: gamma_cf(tt, α, β), t)


def _gamma_sf(x, α, β):
    return jtu.tree_map(lambda xx: gamma_sf(xx, α, β), x)
