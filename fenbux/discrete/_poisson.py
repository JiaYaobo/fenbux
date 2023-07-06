import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jax.dtypes import canonicalize_dtype
from jax.scipy.special import gammainc, gammaln
from jaxtyping import PyTree

from ..base import (
    AbstractDistribution,
    cdf,
    cf,
    DistributionParam,
    domain,
    kurtois,
    mean,
    mgf,
    params,
    ParamType,
    pmf,
    rand,
    skewness,
    standard_dev,
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

    _rate: DistributionParam

    def __init__(self, rate=0.0, dtype=jnp.float_):
        dtype = canonicalize_dtype(dtype)
        self._rate = DistributionParam(
            jtu.tree_map(lambda x: jnp.asarray(x, dtype=dtype), rate)
        )

    @property
    def rate(self):
        return self._rate.val


@eqx.filter_jit
@params.dispatch
def _params(d: Poisson):
    return jtu.tree_leaves(d)


@eqx.filter_jit
@domain.dispatch
def _domain(d: Poisson):
    return jtu.tree_map(lambda _: (0, jnp.inf), d.rate)


@eqx.filter_jit
@mean.dispatch
def _mean(d: Poisson):
    return jtu.tree_map(lambda rate: rate, d.rate)


@eqx.filter_jit
@variance.dispatch
def _variance(d: Poisson):
    return jtu.tree_map(lambda rate: rate, d.rate)


@eqx.filter_jit
@kurtois.dispatch
def _kurtois(d: Poisson):
    return jtu.tree_map(lambda rate: 1 / rate, d.rate)


@eqx.filter_jit
@skewness.dispatch
def _skewness(d: Poisson):
    return jtu.tree_map(lambda rate: 1 / jnp.sqrt(rate), d.rate)


@eqx.filter_jit
@standard_dev.dispatch
def _standard_dev(d: Poisson):
    return jtu.tree_map(lambda rate: jnp.sqrt(rate), d.rate)


@eqx.filter_jit
@pmf.dispatch
def _pmf(d: Poisson, x: ParamType):
    return jtu.tree_map(lambda rate: jnp.exp(_poisson_logpmf(rate, x)), d.rate)


@eqx.filter_jit
@cdf.dispatch
def _cdf(d: Poisson, x: ParamType):
    return jtu.tree_map(lambda rate: _poisson_cdf(rate, x), d.rate)


@eqx.filter_jit
@rand.dispatch
def _rand(d: Poisson, key: PyTree, shape=(), dtype=jnp.int_):
    _key_tree = split_tree(key, d.rate)
    rvs = jtu.tree_map(
        lambda key, r: jr.poisson(key, r, shape=shape, dtype=dtype),
        _key_tree,
        d.rate,
    )
    return rvs


@eqx.filter_jit
@mgf.dispatch
def _mgf(d: Poisson, t: ParamType):
    return jtu.tree_map(lambda rate: _poisson_mgf(rate, t), d.rate)


@eqx.filter_jit
@cf.dispatch
def _cf(d: Poisson, t: ParamType):
    return jtu.tree_map(lambda rate: _poisson_cf(rate, t), d.rate)


def _poisson_cdf(rate, x: ParamType):
    return jtu.tree_map(lambda xx: 1 - gammainc(jnp.floor(xx) + 1, rate), x)


def _poisson_logpmf(rate, x: ParamType):
    return jtu.tree_map(lambda xx: xx * jnp.log(rate) - gammaln(x + 1) - rate, x)


def _poisson_mgf(rate, t: ParamType):
    return jtu.tree_map(lambda tt: jnp.exp(rate * (tt - 1)), t)


def _poisson_cf(rate, t: ParamType):
    return jtu.tree_map(lambda tt: jnp.exp(rate * (1j * tt - 1)), t)
