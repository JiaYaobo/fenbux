import jax.numpy as jnp
import jax.tree_util as jtu
from jax import pure_callback, ShapeDtypeStruct
from jax.scipy.special import gammaln, xlog1py, xlogy
from scipy.stats import binom

from ...core import (
    _check_params_equal_tree_strcutre,
    _intialize_params_tree,
    cdf,
    cf,
    KeyArray,
    kurtosis,
    logcdf,
    logpmf,
    mean,
    mgf,
    params,
    pmf,
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
from ...extension import bdtr, binomial
from ...random_utils import split_tree
from .._base import DiscreteUnivariateDistribution


class Binomial(DiscreteUnivariateDistribution):
    """Binomial distribution.

            X ~ Binomial(n, p)

    Args:
        n (PyTree): Number of trials.
        p (PyTree): Probability of success.
        dtype (jax.numpy.dtype): dtype of the distribution, default jnp.float_.
        use_batch (bool): Whether to use with vmap. Default False.

    Examples:
        >>> import jax.numpy as jnp
        >>> from fenbux import Binomial, logpmf
        >>> dist = Binomial(10, 0.5)
        >>> logpmf(dist, jnp.ones((10, )))
    """

    n: PyTreeVar
    p: PyTreeVar

    def __init__(self, n=0.0, p=0.0, dtype=jnp.float_, use_batch=False):
        _check_params_equal_tree_strcutre(n, p, use_batch=use_batch)
        self.n, self.p = _intialize_params_tree(
            n, p, use_batch=use_batch, dtype=dtype
        )


@params.dispatch
def _params(d: Binomial):
    return (d.n, d.p)


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


@kurtosis.dispatch
def _kurtosis(d: Binomial):
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


@logcdf.dispatch
def _logcdf(d: Binomial, x: PyTreeVar):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda p, n: _binomial_log_cdf(x, p, n), _tree.p, _tree.n)


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

@sf.dispatch
def _sf(d: Binomial, x: PyTreeVar):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda p, n: _binomial_sf(x, p, n), _tree.p, _tree.n)


@rand.dispatch
def _rand(d: Binomial, key: KeyArray, shape: Shape = (), dtype=jnp.int_):
    _tree = d.broadcast_params()
    _key_tree = split_tree(key, _tree.n)
    rvs = jtu.tree_map(
        lambda p, n, k: binomial(k, n, p, shape=shape, dtype=dtype),
        _tree.p,
        _tree.n,
        _key_tree,
    )
    return rvs


def _binomial_log_pmf(x, p, n):
    def _fn(x, p, n):
        k = jnp.floor(x)
        combiln = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
        return combiln + xlogy(k, p) + xlog1py(n - k, -p)

    return jtu.tree_map(lambda xx: _fn(xx, p, n), x)


def _binomial_cdf(x, p, n):
    def _fn(x, p, n):
        k = jnp.floor(x)
        return bdtr(k, n, p)

    return jtu.tree_map(lambda xx: _fn(xx, p, n), x)


def _binomial_log_cdf(x, p, n):
    def _fn(x, p, n):
        k = jnp.floor(x)
        return jnp.log(bdtr(k, n, p))

    return jtu.tree_map(lambda xx: _fn(xx, p, n), x)


def _binomial_mgf(t, p, n):
    def _fn(t, p, n):
        return (1 - p + p * jnp.exp(t)) ** n

    return jtu.tree_map(lambda tt: _fn(tt, p, n), t)


def _binomial_cf(t, p, n):
    def _fn(t, p, n):
        return (1 - p + p * jnp.exp(1j * t)) ** n

    return jtu.tree_map(lambda tt: _fn(tt, p, n), t)


def _binomial_quantile(x, p, n):
    def _fn(x, p, n):
        def _scipy_callback(x, p, n):
            return binom(n, p).ppf(x)

        x = jnp.asarray(x)
        p = jnp.asarray(p)
        n = jnp.asarray(n)
        result_shape_dtype = ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)
        return pure_callback(_scipy_callback, result_shape_dtype, x, p, n)

    return jtu.tree_map(lambda xx: _fn(xx, p, n), x)


def _binomial_sf(x, p, n):
    def _fn(x, p, n):
        k = jnp.floor(x)
        return 1 - bdtr(k, n, p)

    return jtu.tree_map(lambda xx: _fn(xx, p, n), x)
