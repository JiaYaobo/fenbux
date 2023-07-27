import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jax.scipy.special import gammainc, gammaln, polygamma
from jax.scipy.stats.gamma import logpdf as _jax_gamma_logpdf
from tensorflow_probability.substrates.jax.math import igammainv

from ...base import (
    _check_params_equal_tree_strcutre,
    _intialize_params_tree,
    AbstractDistribution,
    cdf,
    cf,
    entropy,
    KeyArray,
    kurtois,
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
from ...random_utils import split_tree


class Gamma(AbstractDistribution):
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
        _check_params_equal_tree_strcutre(shape, rate)
        self.shape, self.rate = _intialize_params_tree(
            shape, rate, use_batch=use_batch, dtype=dtype
        )


@params.dispatch
def _params(d: Gamma):
    return (d.shape, d.rate)


@support.dispatch
def _domain(d: Gamma):
    _tree = d.broadcast_params().shape
    return jtu.tree_map(lambda _: (0.0, jnp.inf), _tree)


@mean.dispatch
def _mean(d: Gamma):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda α, β: α / β, _tree.shape, _tree.rate)


@variance.dispatch
def _variance(d: Gamma):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda α, β: α / (β**2), _tree.shape, _tree.rate)


@standard_dev.dispatch
def _std(d: Gamma):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda α, β: jnp.sqrt(α / (β**2)), _tree.shape, _tree.rate)


@kurtois.dispatch
def _kurtois(d: Gamma):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda α: 6 / α, _tree.shape)


@skewness.dispatch
def _skewness(d: Gamma):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda α: 2 / jnp.sqrt(α), _tree.shape)


@entropy.dispatch
def _entropy(d: Gamma):
    _tree = d.broadcast_params()
    return jtu.tree_map(
        lambda α, β: α - jnp.log(β) + gammaln(α) + (1 - α) * polygamma(1, α),
        _tree.shape,
        _tree.rate,
    )


@logpdf.dispatch
def _logpdf(d: Gamma, x: PyTreeVar):
    _tree = d.broadcast_params()
    log_d = jtu.tree_map(lambda α, β: _gamma_log_pdf(x, α, β), _tree.shape, _tree.rate)
    return log_d


@pdf.dispatch
def _pdf(d: Gamma, x: PyTreeVar):
    _logpdf = logpdf(d, x)
    return jtu.tree_map(lambda log_d: jnp.exp(log_d), _logpdf)


@logcdf.dispatch
def _logcdf(d: Gamma, x: PyTreeVar):
    _tree = d.broadcast_params()
    log_cdf = jtu.tree_map(
        lambda α, β: _gamma_log_cdf(x, α, β), _tree.shape, _tree.rate
    )
    return log_cdf


@cdf.dispatch
def _cdf(d: Gamma, x: PyTreeVar):
    _tree = d.broadcast_params()
    prob = jtu.tree_map(lambda α, β: _gamma_cdf(x, α, β), _tree.shape, _tree.rate)
    return prob


@quantile.dispatch
def _quantile(d: Gamma, q: PyTreeVar):
    _tree = d.broadcast_params()
    x = jtu.tree_map(lambda α, β: _gamma_quantile(q, α, β), _tree.shape, _tree.rate)
    return x


@rand.dispatch
def _rand(d: Gamma, key: KeyArray, shape: Shape = (), dtype=jnp.float_):
    _tree = d.broadcast_params()
    _key_tree = split_tree(key, _tree.shape)
    rvs = jtu.tree_map(
        lambda α, β, key: jr.gamma(key, shape, dtype=dtype) * β + α,
        _tree.shape,
        _tree.rate,
        _key_tree,
    )
    return rvs


@mgf.dispatch
def _mgf(d: Gamma, t: PyTreeVar):
    _tree = d.broadcast_params()
    mgf = jtu.tree_map(lambda α, β: _gamma_mgf(t, α, β), _tree.shape, _tree.rate)
    return mgf


@cf.dispatch
def _cf(d: Gamma, t: PyTreeVar):
    _tree = d.broadcast_params()
    cf = jtu.tree_map(lambda α, β: _gamma_cf(t, α, β), _tree.shape, _tree.rate)
    return cf


@sf.dispatch
def _sf(d: Gamma, x: PyTreeVar):
    _tree = d.broadcast_params()
    sf = jtu.tree_map(lambda α, β: _gamma_sf(x, α, β), _tree.shape, _tree.rate)
    return sf


def _gamma_log_pdf(x, α, β):
    def _fn(x, α, β):
        return _jax_gamma_logpdf(x, a=α, scale=1 / β)

    return jtu.tree_map(lambda xx: _fn(xx, α, β), x)


def _gamma_cdf(x, α, β):
    def _fn(x, α, β):
        return gammainc(α, x * β)

    return jtu.tree_map(lambda xx: _fn(xx, α, β), x)


def _gamma_quantile(x, α, β):
    def _fn(x, α, β):
        return igammainv(α, x) / β

    return jtu.tree_map(lambda xx: _fn(xx, α, β), x)


def _gamma_mgf(t, α, β):
    def _fn(t, α, β):
        return (1 - t / β) ** (-α)

    return jtu.tree_map(lambda tt: _fn(tt, α, β), t)


def _gamma_cf(t, α, β):
    def _fn(t, α, β):
        return (1 - 1j * t / β) ** (-α)

    return jtu.tree_map(lambda tt: _fn(tt, α, β), t)


def _gamma_sf(x, α, β):
    def _fn(x, α, β):
        return 1 - gammainc(α, x * β)

    return jtu.tree_map(lambda xx: _fn(xx, α, β), x)


def _gamma_log_cdf(x, α, β):
    def _fn(x, α, β):
        return jnp.log(gammainc(α, x * β))

    return jtu.tree_map(lambda xx: _fn(xx, α, β), x)
