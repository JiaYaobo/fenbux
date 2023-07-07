import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jax.dtypes import canonicalize_dtype

from ..base import (
    AbstractDistribution,
    cdf,
    cf,
    DistributionParam,
    entropy,
    KeyArray,
    kurtois,
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


class Bernoulli(AbstractDistribution):
    """Bernoulli distribution.
        X ~ Bernoulli(p)
    Args:
        p (ArrayLike): Probability of success.
        dtype (jax.numpy.dtype): dtype of the distribution, default jnp.float_.
    """

    _p: DistributionParam

    def __init__(self, p=0.0, dtype=jnp.float_):
        dtype = canonicalize_dtype(dtype)
        self._p = DistributionParam(
            jtu.tree_map(lambda x: jnp.asarray(x, dtype=dtype), p)
        )

    @property
    def p(self):
        return self._p.val


@params.dispatch
def _params(d: Bernoulli):
    return jtu.tree_leaves(d)


@support.dispatch
def _domain(d: Bernoulli):
    return jtu.tree_map(lambda _: {0, 1}, d.p)


@eqx.filter_jit
@mean.dispatch
def _mean(d: Bernoulli):
    return jtu.tree_map(lambda p: p, d.p)


@eqx.filter_jit
@variance.dispatch
def _variance(d: Bernoulli):
    return jtu.tree_map(lambda p: p * (1 - p), d.p)


@eqx.filter_jit
@standard_dev.dispatch
def _standard_dev(d: Bernoulli):
    return jtu.tree_map(lambda p: jnp.sqrt(p * (1 - p)), d.p)


@eqx.filter_jit
@kurtois.dispatch
def _kurtois(d: Bernoulli):
    return jtu.tree_map(lambda p: (1 - 6 * p * (1 - p)) / (p * (1 - p)), d.p)


@eqx.filter_jit
@skewness.dispatch
def _skewness(d: Bernoulli):
    return jtu.tree_map(lambda p: (1 - 2 * p) / jnp.sqrt(p * (1 - p)), d.p)


@eqx.filter_jit
@entropy.dispatch
def _entropy(d: Bernoulli):
    return jtu.tree_map(lambda p: -p * jnp.log(p) - (1 - p) * jnp.log(1 - p), d.p)


@eqx.filter_jit
@pmf.dispatch
def _pmf(d: Bernoulli, x: PyTreeVar):
    return jtu.tree_map(lambda p: _bernoulli_pmf(p, x), d.p)


@eqx.filter_jit
@rand.dispatch
def _rand(d: Bernoulli, key: KeyArray, shape: Shape = (), dtype=jnp.float_):
    _key_tree = split_tree(key, d.p)
    rvs = jtu.tree_map(
        lambda p, k: jr.bernoulli(k, p, shape=shape, dtype=dtype),
        d.p,
        _key_tree,
    )
    return rvs


@eqx.filter_jit
@cdf.dispatch
def _cdf(d: Bernoulli, x: PyTreeVar):
    return jtu.tree_map(lambda p: _bernoulli_cdf(p, x), d.p)


@eqx.filter_jit
@quantile.dispatch
def _quantile(d: Bernoulli, x: PyTreeVar):
    return jtu.tree_map(lambda p: _bernoulli_quantile(p, x), d.p)


@eqx.filter_jit
@mgf.dispatch
def _mgf(d: Bernoulli, t: PyTreeVar):
    return jtu.tree_map(lambda p: _bernoulli_mgf(p, t), d.p)


@eqx.filter_jit
@cf.dispatch
def _cf(d: Bernoulli, t: PyTreeVar):
    return jtu.tree_map(lambda p: _bernoulli_cf(p, t), d.p)


def _bernoulli_pmf(p, x):
    return jtu.tree_map(lambda xx: p**xx * (1 - p) ** (1 - xx), x)


def _bernoulli_cdf(p, x):
    return jtu.tree_map(lambda xx: jnp.where(xx >= 1.0, 1.0, 1.0 - p), x)


def _bernoulli_quantile(p, x):
    return jtu.tree_map(lambda xx: jnp.where(xx > p, 1.0, 0.0), x)


def _bernoulli_mgf(p, t):
    return jtu.tree_map(lambda tt: 1 - p + p * jnp.exp(tt), t)


def _bernoulli_cf(p, t):
    return jtu.tree_map(lambda tt: 1 - p + p * jnp.exp(1j * tt), t)
