import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jax.dtypes import canonicalize_dtype
from jax.scipy.special import gammainc, gammaln, polygamma
from jax.scipy.stats.gamma import logpdf as _jax_gamma_logpdf
from tensorflow_probability.substrates.jax.math import igammainv

from ..base import (
    AbstractDistribution,
    cdf,
    cf,
    DistributionParam,
    entropy,
    KeyArray,
    kurtois,
    logpdf,
    mean,
    mgf,
    params,
    pdf,
    PyTreeVar,
    quantile,
    rand,
    Shape,
    skewness,
    standard_dev,
    support,
    variance,
)
from ..random_utils import split_tree


class Gamma(AbstractDistribution):
    """Gamma distribution.

        X ~ Gamma(shape, rate)

    Args:
        shape (ArrayLike): Shape parameter of the distribution.
        rate (ArrayLike): Rate parameter of the distribution.
        dtype (jax.numpy.dtype): dtype of the distribution, default jnp.float_.
    """

    _shape: DistributionParam
    _rate: DistributionParam

    def __init__(self, shape=0.0, rate=0.0, dtype=jnp.float_):
        if jtu.tree_structure(shape) != jtu.tree_structure(rate):
            raise ValueError(
                f"shape and rate must have the same tree structure, got {jtu.tree_structure(shape)} and {jtu.tree_structure(rate)}"
            )

        dtype = canonicalize_dtype(dtype)
        self._shape = DistributionParam(
            jtu.tree_map(lambda x: jnp.asarray(x, dtype=dtype), shape)
        )
        self._rate = DistributionParam(
            jtu.tree_map(lambda x: jnp.asarray(x, dtype=dtype), rate)
        )

    @property
    def shape(self):
        return self._shape.val

    @property
    def rate(self):
        return self._rate.val


@params.dispatch
def _params(d: Gamma):
    return jtu.tree_leaves(d)


@support.dispatch
def _domain(d: Gamma):
    _tree = d.broadcast_params().shape
    return jtu.tree_map(lambda _: (0.0, jnp.inf), _tree)


@eqx.filter_jit
@mean.dispatch
def _mean(d: Gamma):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda α, β: α / β, _tree.shape, _tree.rate)


@eqx.filter_jit
@variance.dispatch
def _variance(d: Gamma):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda α, β: α / (β**2), _tree.shape, _tree.rate)


@eqx.filter_jit
@standard_dev.dispatch
def _std(d: Gamma):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda α, β: jnp.sqrt(α / (β**2)), _tree.shape, _tree.rate)


@eqx.filter_jit
@kurtois.dispatch
def _kurtois(d: Gamma):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda α: 6 / α, _tree.shape)


@eqx.filter_jit
@skewness.dispatch
def _skewness(d: Gamma):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda α: 2 / jnp.sqrt(α), _tree.shape)


@eqx.filter_jit
@entropy.dispatch
def _entropy(d: Gamma):
    _tree = d.broadcast_params()
    return jtu.tree_map(
        lambda α, β: α - jnp.log(β) + gammaln(α) + (1 - α) * polygamma(1, α),
        _tree.shape,
        _tree.rate,
    )


@eqx.filter_jit
@logpdf.dispatch
def _logpdf(d: Gamma, x: PyTreeVar):
    _tree = d.broadcast_params()
    log_d = jtu.tree_map(lambda α, β: _gamma_log_pdf(x, α, β), _tree.shape, _tree.rate)
    return log_d


@eqx.filter_jit
@pdf.dispatch
def _pdf(d: Gamma, x: PyTreeVar):
    _logpdf = logpdf(d, x)
    return jtu.tree_map(lambda log_d: jnp.exp(log_d), _logpdf)


@eqx.filter_jit
@cdf.dispatch
def _cdf(d: Gamma, x: PyTreeVar):
    _tree = d.broadcast_params()
    prob = jtu.tree_map(lambda α, β: _gamma_cdf(x, α, β), _tree.shape, _tree.rate)
    return prob


@eqx.filter_jit
@quantile.dispatch
def _quantile(d: Gamma, q: PyTreeVar):
    _tree = d.broadcast_params()
    x = jtu.tree_map(lambda α, β: _gamma_quantile(q, α, β), _tree.shape, _tree.rate)
    return x


@eqx.filter_jit
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


@eqx.filter_jit
@mgf.dispatch
def _mgf(d: Gamma, t: PyTreeVar):
    _tree = d.broadcast_params()
    mgf = jtu.tree_map(lambda α, β: _gamma_mgf(t, α, β), _tree.shape, _tree.rate)
    return mgf


@eqx.filter_jit
@cf.dispatch
def _cf(d: Gamma, t: PyTreeVar):
    _tree = d.broadcast_params()
    cf = jtu.tree_map(lambda α, β: _gamma_cf(t, α, β), _tree.shape, _tree.rate)
    return cf


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
