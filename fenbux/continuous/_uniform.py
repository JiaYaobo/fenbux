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
from ..tree_utils import full_pytree


class Uniform(AbstractDistribution):
    """Uniform distribution.
        X ~ Uniform(lower, upper)
    Args:
        lower (ArrayLike): Lower bound of the distribution.
        upper (ArrayLike): Upper bound of the distribution.
        dtype (jax.numpy.dtype): dtype of the distribution, default jnp.float_.
    """

    _lower: DistributionParam
    _upper: DistributionParam

    def __init__(
        self, lower: PyTreeVar = 0.0, upper: PyTreeVar = 1.0, dtype=jnp.float_
    ):
        if jtu.tree_structure(lower) != jtu.tree_structure(upper):
            raise ValueError(
                f"lower and upper must have the same tree structure, got {jtu.tree_structure(lower)} and {jtu.tree_structure(upper)}"
            )
        dtype = canonicalize_dtype(dtype)
        self._lower = DistributionParam(
            jtu.tree_map(lambda x: jnp.asarray(x, dtype=dtype), lower)
        )
        self._upper = DistributionParam(
            jtu.tree_map(lambda x: jnp.asarray(x, dtype=dtype), upper)
        )

    @property
    def lower(self):
        return self._lower.val

    @property
    def upper(self):
        return self._upper.val


@params.dispatch
def _params(d: Uniform):
    return jtu.tree_leaves(d)


@eqx.filter_jit
@support.dispatch
def _domain(d: Uniform):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda l, u: (l, u), _tree.lower, _tree.upper)


@eqx.filter_jit
@mean.dispatch
def _mean(d: Uniform):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda l, u: (l + u) / 2, _tree.lower, _tree.upper)


@eqx.filter_jit
@variance.dispatch
def _variance(d: Uniform):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda l, u: (u - l) ** 2 / 12, _tree.lower, _tree.upper)


@eqx.filter_jit
@standard_dev.dispatch
def _standard_dev(d: Uniform):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda l, u: (u - l) / jnp.sqrt(12), _tree.lower, _tree.upper)


@eqx.filter_jit
@kurtois.dispatch
def _kurtois(d: Uniform):
    shape = d.broadcast_shapes()
    return full_pytree(shape, -6 / 5).val


@eqx.filter_jit
@skewness.dispatch
def _skewness(d: Uniform):
    shape = d.broadcast_shapes()
    return full_pytree(shape, 0.0).val


@eqx.filter_jit
@entropy.dispatch
def _entropy(d: Uniform):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda l, u: jnp.log(u - l), _tree.lower, _tree.upper)


@eqx.filter_jit
@rand.dispatch
def _rand(d: Uniform, key: KeyArray, shape: Shape = (), dtype=jnp.float_):
    _tree = d.broadcast_params()
    lower, upper = _tree.lower, _tree.upper
    _key_tree = split_tree(key, _tree.lower)
    return jtu.tree_map(
        lambda l, u, k: jr.uniform(k, shape, dtype) * (u - l) + l,
        lower,
        upper,
        _key_tree,
    )


@eqx.filter_jit
@quantile.dispatch
def _quantile(d: Uniform, x: PyTreeVar):
    _tree = d.broadcast_params()
    return jtu.tree_map(
        lambda l, u: _uniform_quantile(x, l, u), _tree.lower, _tree.upper
    )


@eqx.filter_jit
@pdf.dispatch
def _pdf(d: Uniform, x: PyTreeVar):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda l, u: _uniform_pdf(x, l, u), _tree.lower, _tree.upper)


@eqx.filter_jit
@logpdf.dispatch
def _logpdf(d: Uniform, x: PyTreeVar):
    _tree = d.broadcast_params()
    return jtu.tree_map(
        lambda l, u: _uniform_log_pdf(x, l, u), _tree.lower, _tree.upper
    )


@eqx.filter_jit
@cdf.dispatch
def _cdf(d: Uniform, x: PyTreeVar):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda l, u: _uniform_cdf(x, l, u), _tree.lower, _tree.upper)


@eqx.filter_jit
@mgf.dispatch
def _mgf(d: Uniform, t: PyTreeVar):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda l, u: _uniform_mgf(t, l, u), _tree.lower, _tree.upper)


@eqx.filter_jit
@cf.dispatch
def _cf(d: Uniform, t: PyTreeVar):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda l, u: _uniform_cf(t, l, u), _tree.lower, _tree.upper)


@eqx.filter_jit
@sf.dispatch
def _sf(d: Uniform, x: PyTreeVar):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda l, u: _uniform_sf(x, l, u), _tree.lower, _tree.upper)


def _uniform_log_pdf(x, lower, upper):
    return jtu.tree_map(lambda x: -jnp.log(upper - lower), x)


def _uniform_pdf(x, lower, upper):
    return jtu.tree_map(lambda xx: 1 / (upper - lower), x)


def _uniform_cdf(x, lower, upper):
    return jtu.tree_map(lambda xx: (xx - lower) / (upper - lower), x)


def _uniform_quantile(x, lower, upper):
    return jtu.tree_map(lambda xx: xx * (upper - lower) + lower, x)


def _uniform_mgf(t, lower, upper):
    return jtu.tree_map(
        lambda tt: (jnp.exp(tt * upper) - jnp.exp(tt * lower)) / (tt * (upper - lower)),
        t,
    )


def _uniform_cf(t, lower, upper):
    return jtu.tree_map(
        lambda tt: (jnp.exp(1j * tt * upper) - jnp.exp(1j * tt * lower))
        / (1j * tt * (upper - lower)),
        t,
    )


def _uniform_sf(x, lower, upper):
    return jtu.tree_map(lambda xx: 1 - (xx - lower) / (upper - lower), x)
