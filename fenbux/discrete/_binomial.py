import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jax.dtypes import canonicalize_dtype
from jax.scipy.special import betainc, gammaln
from tensorflow_probability.substrates.jax.math import igammainv

from ..base import (
    AbstractDistribution,
    cdf,
    cf,
    kurtois,
    logpmf,
    mean,
    mgf,
    params,
    pmf,
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


class Binomial(AbstractDistribution):
    p: PyTreeVar
    n: PyTreeVar

    def __init__(self, p=0.0, n=0.0, dtype=jnp.float_):
        dtype = canonicalize_dtype(dtype)
        self.p = jtu.tree_map(lambda x: jnp.asarray(x, dtype=dtype), p)
        self.n = jtu.tree_map(lambda x: jnp.asarray(x, dtype=dtype), n)


@params.dispatch
def _params(d: Binomial):
    return jtu.tree_leaves(d)


@support.dispatch
def _domain(d: Binomial):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda n: {*[nn for nn in range(n)]}, _tree.n)


@mean.dispatch
def _mean(d: Binomial):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda p, n: p * n, _tree.p, _tree.n)


@variance.dispatch
def _variance(d: Binomial):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda p, n: p * (1 - p) * n, _tree.p, _tree.n)


@standard_dev.dispatch
def _standard_dev(d: Binomial):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda p, n: jnp.sqrt(p * (1 - p) * n), _tree.p, _tree.n)


@skewness.dispatch
def _skewness(d: Binomial):
    _tree = d.broadcast_params()
    return jtu.tree_map(
        lambda p, n: (1 - 2 * p) / jnp.sqrt(p * (1 - p) * n), _tree.p, _tree.n
    )


@kurtois.dispatch
def _kurtois(d: Binomial):
    _tree = d.broadcast_params()
    return jtu.tree_map(
        lambda p, n: (1 - 6 * p * (1 - p)) / (p * (1 - p) * n), _tree.p, _tree.n
    )


@logpmf.dispatch
def _logpmf(d: Binomial, x: PyTreeVar):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda p, n: _binomial_log_pmf(x, p, n), _tree.p, _tree.n)


@pmf.dispatch
def _pmf(d: Binomial, x: PyTreeVar):
    _tree = d.broadcast_params()
    log_pmf = jtu.tree_map(lambda p, n: _binomial_log_pmf(x, p, n), _tree.p, _tree.n)
    return jtu.tree_map(lambda _log_pmf: jnp.exp(_log_pmf), log_pmf)


@cdf.dispatch
def _cdf(d: Binomial, x: PyTreeVar):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda p, n: _binomial_cdf(x, p, n), _tree.p, _tree.n)


@quantile.dispatch
def _quantile(d: Binomial, q: PyTreeVar):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda p, n: _binomial_quantile(q, p, n), _tree.p, _tree.n)


@mgf.dispatch
def _mgf(d: Binomial, t: PyTreeVar):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda p, n: _binomial_mgf(t, p, n), _tree.p, _tree.n)


@cf.dispatch
def _cf(d: Binomial, t: PyTreeVar):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda p, n: _binomial_cf(t, p, n), _tree.p, _tree.n)


def _binomial_log_pmf(x, p, n):
    def _fn(x, p, n):
        return (
            gammaln(n + 1)
            - gammaln(x + 1)
            - gammaln(n - x + 1)
            + x * jnp.log(p)
            + (n - x) * jnp.log(1 - p)
        )

    return jtu.tree_map(lambda xx: _fn(xx, p, n), x)


def _binomial_cdf(x, p, n):
    def _fn(x, p, n):
        return betainc(x + 1, n - x + 1, p)

    return jtu.tree_map(lambda xx: _fn(xx, p, n), x)


def _binomial_mgf(t, p, n):
    def _fn(t, p, n):
        return (1 - p + p * jnp.exp(t)) ** n

    return jtu.tree_map(lambda tt: _fn(tt, p, n), t)


def _binomial_cf(t, p, n):
    def _fn(t, p, n):
        return (1 - p + p * jnp.exp(1j * t)) ** n

    return jtu.tree_map(lambda tt: _fn(tt, p, n), t)


def _binomial_quantile(q, p, n):
    def _fn(q, p, n):
        return igammainv(n - q, n - p, p)

    return jtu.tree_map(lambda qq: _fn(qq, p, n), q)
