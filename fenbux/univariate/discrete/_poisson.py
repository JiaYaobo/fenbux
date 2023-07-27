import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jax.scipy.special import gammainc, gammaln

from ...base import (
    _intialize_params_tree,
    AbstractDistribution,
    cdf,
    cf,
    KeyArray,
    kurtois,
    mean,
    mgf,
    params,
    pmf,
    PyTreeVar,
    rand,
    Shape,
    skewness,
    standard_dev,
    support,
    variance,
)
from ...random_utils import split_tree


class Poisson(AbstractDistribution):
    """Poisson distribution.

        X ~ Poisson(λ)

    Args:
        rate (PyTree): Rate parameter of the distribution.
        dtype (jax.numpy.dtype): dtype of the distribution, default jnp.float_.
        use_batch (bool): Whether to use with vmap. Default False.

    Examples:
        >>> import jax.numpy as jnp
        >>> from fenbux import Poisson, logpdf
        >>> dist = Poisson(1.0)
        >>> logpdf(dist, jnp.ones((10, )))
    """

    rate: PyTreeVar

    def __init__(self, rate=0.0, dtype=jnp.float_, use_batch=False):
        self.rate = _intialize_params_tree(rate, use_batch=use_batch, dtype=dtype)


@params.dispatch
def _params(d: Poisson):
    return (d.rate,)


@support.dispatch
def _domain(d: Poisson):
    return jtu.tree_map(lambda _: (0, jnp.inf), d.rate)


@mean.dispatch
def _mean(d: Poisson):
    return jtu.tree_map(lambda rate: rate, d.rate)


@variance.dispatch
def _variance(d: Poisson):
    return jtu.tree_map(lambda rate: rate, d.rate)


@kurtois.dispatch
def _kurtois(d: Poisson):
    return jtu.tree_map(lambda rate: 1 / rate, d.rate)


@skewness.dispatch
def _skewness(d: Poisson):
    return jtu.tree_map(lambda rate: 1 / jnp.sqrt(rate), d.rate)


@standard_dev.dispatch
def _standard_dev(d: Poisson):
    return jtu.tree_map(lambda rate: jnp.sqrt(rate), d.rate)


@pmf.dispatch
def _pmf(d: Poisson, x: PyTreeVar):
    return jtu.tree_map(lambda rate: _poisson_pmf(rate, x), d.rate)


@cdf.dispatch
def _cdf(d: Poisson, x: PyTreeVar):
    return jtu.tree_map(lambda rate: _poisson_cdf(rate, x), d.rate)


@rand.dispatch
def _rand(d: Poisson, key: KeyArray, shape: Shape = (), dtype=jnp.int_):
    _key_tree = split_tree(key, d.rate)
    rvs = jtu.tree_map(
        lambda key, r: jr.poisson(key, r, shape=shape, dtype=dtype),
        _key_tree,
        d.rate,
    )
    return rvs


@mgf.dispatch
def _mgf(d: Poisson, t: PyTreeVar):
    return jtu.tree_map(lambda rate: _poisson_mgf(rate, t), d.rate)


@cf.dispatch
def _cf(d: Poisson, t: PyTreeVar):
    return jtu.tree_map(lambda rate: _poisson_cf(rate, t), d.rate)


def _poisson_cdf(rate, x: PyTreeVar):
    return jtu.tree_map(lambda xx: 1 - gammainc(jnp.floor(xx) + 1, rate), x)


def _poisson_logpmf(rate, x: PyTreeVar):
    def _fn(r, x):
        return x * jnp.log(r) - gammaln(x + 1) - r

    return jtu.tree_map(lambda xx: _fn(rate, xx), x)


def _poisson_pmf(rate, x: PyTreeVar):
    def _fn(r, x):
        return jnp.exp(x * jnp.log(r) - gammaln(x + 1) - r)

    return jtu.tree_map(lambda xx: _fn(rate, xx), x)


def _poisson_mgf(rate, t: PyTreeVar):
    return jtu.tree_map(lambda tt: jnp.exp(rate * (tt - 1)), t)


def _poisson_cf(rate, t: PyTreeVar):
    return jtu.tree_map(lambda tt: jnp.exp(rate * (1j * tt - 1)), t)
