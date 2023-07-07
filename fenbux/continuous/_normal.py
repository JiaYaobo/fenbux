import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jax.dtypes import canonicalize_dtype
from jax.scipy.special import ndtr, ndtri

from ..base import (
    AbstractDistribution,
    cdf,
    cf,
    DistributionParam,
    DTypeLikeFloat,
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
    sf,
    Shape,
    skewness,
    standard_dev,
    support,
    variance,
)
from ..random_utils import split_tree
from ..tree_utils import zeros_pytree


class Normal(AbstractDistribution):
    """Normal distribution.
        X ~ Normal(μ, σ)
    Args:
        mean (ArrayLike): Mean of the distribution.
        sd (ArrayLike): Standard deviation of the distribution.
        dtype (jax.numpy.dtype): dtype of the distribution, default jnp.float_.
    """

    _mean: DistributionParam
    _sd: DistributionParam

    def __init__(self, mean=0.0, sd=0.0, dtype=jnp.float_):
        if jtu.tree_structure(mean) != jtu.tree_structure(sd):
            raise ValueError(
                f"mean and sd must have the same tree structure, got {jtu.tree_structure(mean)} and {jtu.tree_structure(sd)}"
            )

        dtype = canonicalize_dtype(dtype)
        self._mean = DistributionParam(
            jtu.tree_map(lambda x: jnp.asarray(x, dtype=dtype), mean)
        )
        self._sd = DistributionParam(
            jtu.tree_map(lambda x: jnp.asarray(x, dtype=dtype), sd)
        )

    @property
    def mean(self):
        return self._mean.val

    @property
    def sd(self):
        return self._sd.val


@eqx.filter_jit
@params.dispatch
def _params(d: Normal):
    return jtu.tree_leaves(d)


@eqx.filter_jit
@support.dispatch
def _domain(d: Normal):
    _tree = d.broadcast_params().mean
    return jtu.tree_map(lambda _: (jnp.NINF, jnp.inf), _tree)


@eqx.filter_jit
@mean.dispatch
def _mean(d: Normal):
    return d.broadcast_params().mean


@eqx.filter_jit
@variance.dispatch
def _variance(d: Normal):
    return jtu.tree_map(lambda x: x**2, d.broadcast_params().sd)


@eqx.filter_jit
@standard_dev.dispatch
def _std(d: Normal):
    return d.broadcast_params().sd


@eqx.filter_jit
@kurtois.dispatch
def _kurtois(d: Normal):
    shape = d.broadcast_shapes()
    return zeros_pytree(shape).val


@eqx.filter_jit
@skewness.dispatch
def _skewness(d: Normal):
    shape = d.broadcast_shapes()
    return zeros_pytree(shape).val


@eqx.filter_jit
@entropy.dispatch
def _entropy(d: Normal):
    _tree = d.broadcast_params()
    entropy = jtu.tree_map(lambda σ: 0.5 * jnp.log(2 * jnp.pi * σ**2) + 0.5, _tree.sd)
    return entropy


@eqx.filter_jit
@logpdf.dispatch
def _logpdf(d: Normal, x: PyTreeVar):
    _tree = d.broadcast_params()
    log_d = jtu.tree_map(lambda μ, σ: _normal_log_pdf(x, μ, σ), _tree.mean, _tree.sd)
    return log_d


@eqx.filter_jit
@pdf.dispatch
def _pdf(d: Normal, x: PyTreeVar):
    _tree = d.broadcast_params()
    d = jtu.tree_map(lambda μ, σ: _normal_pdf(x, μ, σ), _tree.mean, _tree.sd)
    return d


@eqx.filter_jit
@cdf.dispatch
def _cdf(d: Normal, x: PyTreeVar):
    _tree = d.broadcast_params()
    prob = jtu.tree_map(lambda μ, σ: _normal_cdf(x, μ, σ), _tree.mean, _tree.sd)
    return prob


@eqx.filter_jit
@quantile.dispatch
def _quantile(d: Normal, q: PyTreeVar):
    _tree = d.broadcast_params()
    x = jtu.tree_map(lambda μ, σ: _normal_quantile(q, μ, σ), _tree.mean, _tree.sd)
    return x


@eqx.filter_jit
@rand.dispatch
def _rand(
    d: Normal, key: KeyArray, shape: Shape = (), dtype: DTypeLikeFloat = jnp.float_
):
    _tree = d.broadcast_params()
    _key_tree = split_tree(key, _tree.mean)
    rvs = jtu.tree_map(
        lambda μ, σ, key: jr.normal(key, shape, dtype=dtype) * σ + μ,
        _tree.mean,
        _tree.sd,
        _key_tree,
    )
    return rvs


@eqx.filter_jit
@mgf.dispatch
def _mgf(d: Normal, t: PyTreeVar):
    _tree = d.broadcast_params()
    mgf = jtu.tree_map(lambda μ, σ: _normal_mgf(t, μ, σ), _tree.mean, _tree.sd)
    return mgf


@eqx.filter_jit
@cf.dispatch
def _cf(d: Normal, t: PyTreeVar):
    _tree = d.broadcast_params()
    cf = jtu.tree_map(lambda μ, σ: _normal_cf(t, μ, σ), _tree.mean, _tree.sd)
    return cf


@sf.dispatch
def _sf(d: Normal, x: PyTreeVar):
    _tree = d.broadcast_params()
    sf = jtu.tree_map(lambda μ, σ: _normal_sf(x, μ, σ), _tree.mean, _tree.sd)
    return sf


def _normal_cf(t, μ, σ):
    def _fn(t, μ, σ):
        return jnp.exp(1j * μ * t - 0.5 * σ**2 * t**2)

    return jtu.tree_map(lambda tt: _fn(tt, μ, σ), t)


def _normal_mgf(t, μ, σ):
    def _fn(t, μ, σ):
        return jnp.exp(μ * t + 0.5 * σ**2 * t**2)

    return jtu.tree_map(lambda tt: _fn(tt, μ, σ), t)


def _normal_pdf(x, μ, σ):
    def _fn(x, μ, σ):
        return jnp.exp(-((x - μ) ** 2) / (2 * σ**2)) / (σ * jnp.sqrt(2 * jnp.pi))

    return jtu.tree_map(lambda xx: _fn(xx, μ, σ), x)


def _normal_log_pdf(x, μ, σ):
    def _fn(x, μ, σ):
        return -((x - μ) ** 2) / (2 * σ**2) - jnp.log(σ * jnp.sqrt(2 * jnp.pi))

    return jtu.tree_map(lambda xx: _fn(xx, μ, σ), x)


def _normal_cdf(x, μ, σ):
    def _fn(x, μ, σ):
        return ndtr((x - μ) / σ)

    return jtu.tree_map(lambda xx: _fn(xx, μ, σ), x)


def _normal_quantile(x, μ, σ):
    def _fn(x, μ, σ):
        return ndtri(x) * σ + μ

    return jtu.tree_map(lambda xx: _fn(xx, μ, σ), x)


def _normal_sf(x, μ, σ):
    def _fn(x, μ, σ):
        return 1 - ndtr((x - μ) / σ)

    return jtu.tree_map(lambda xx: _fn(xx, μ, σ), x)
