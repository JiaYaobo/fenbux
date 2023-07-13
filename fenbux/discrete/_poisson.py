import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jax.dtypes import canonicalize_dtype
from jax.scipy.special import gammainc, gammaln

from ..base import (
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
from ..random_utils import split_tree


class Poisson(AbstractDistribution):
    """Poisson distribution.

        X ~ Poisson(λ)

    Args:
        rate (ArrayLike): Rate parameter of the distribution.
        dtype (jax.numpy.dtype): dtype of the distribution, default jnp.float_.
    """

    rate: PyTreeVar

    def __init__(self, rate=0.0, dtype=jnp.float_):
        dtype = canonicalize_dtype(dtype)
        self.rate = jtu.tree_map(lambda x: jnp.asarray(x, dtype=dtype), rate)


@params.dispatch
def _params(d: Poisson):
    return jtu.tree_leaves(d)


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
    return jtu.tree_map(lambda rate: jnp.exp(_poisson_logpmf(rate, x)), d.rate)


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
    return jtu.tree_map(lambda xx: xx * jnp.log(rate) - gammaln(x + 1) - rate, x)


def _poisson_mgf(rate, t: PyTreeVar):
    return jtu.tree_map(lambda tt: jnp.exp(rate * (tt - 1)), t)


def _poisson_cf(rate, t: PyTreeVar):
    return jtu.tree_map(lambda tt: jnp.exp(rate * (1j * tt - 1)), t)
