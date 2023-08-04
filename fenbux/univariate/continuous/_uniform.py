import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

from ...core import (
    _check_params_equal_tree_strcutre,
    _intialize_params_tree,
    cdf,
    cf,
    entropy,
    KeyArray,
    kurtosis,
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
from ...tree_utils import full_pytree
from .._base import ContinuousUnivariateDistribution


class Uniform(ContinuousUnivariateDistribution):
    """Uniform distribution.
        X ~ Uniform(lower, upper)
    Args:
        lower (PyTree): Lower bound of the distribution.
        upper (PyTree): Upper bound of the distribution.
        dtype (jax.numpy.dtype): dtype of the distribution, default jnp.float_.
        use_batch (bool): Whether to use with vmap. Default False.

    Examples:
        >>> import jax.numpy as jnp
        >>> from fenbux import Uniform, logpdf
        >>> dist = Uniform(0.0, 1.0)
        >>> logpdf(dist, jnp.ones((10, )))
    """

    lower: PyTreeVar
    upper: PyTreeVar

    def __init__(
        self,
        lower: PyTreeVar = 0.0,
        upper: PyTreeVar = 1.0,
        dtype=jnp.float_,
        use_batch=False,
    ):
        _check_params_equal_tree_strcutre(lower, upper)
        self.lower, self.upper = _intialize_params_tree(
            lower, upper, use_batch=use_batch, dtype=dtype
        )


@params.dispatch
def _params(d: Uniform):
    return (d.lower, d.upper)


@support.dispatch
def _domain(d: Uniform):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda l, u: (l, u), _tree.lower, _tree.upper)


@mean.dispatch
def _mean(d: Uniform):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda l, u: (l + u) / 2, _tree.lower, _tree.upper)


@variance.dispatch
def _variance(d: Uniform):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda l, u: (u - l) ** 2 / 12, _tree.lower, _tree.upper)


@standard_dev.dispatch
def _standard_dev(d: Uniform):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda l, u: (u - l) / jnp.sqrt(12), _tree.lower, _tree.upper)


@kurtosis.dispatch
def _kurtosis(d: Uniform):
    shape = d.broadcast_shapes()
    return full_pytree(shape, -6 / 5)


@skewness.dispatch
def _skewness(d: Uniform):
    shape = d.broadcast_shapes()
    return full_pytree(shape, 0.0)


@entropy.dispatch
def _entropy(d: Uniform):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda l, u: jnp.log(u - l), _tree.lower, _tree.upper)


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


@quantile.dispatch
def _quantile(d: Uniform, x: PyTreeVar):
    _tree = d.broadcast_params()
    return jtu.tree_map(
        lambda l, u: _uniform_quantile(x, l, u), _tree.lower, _tree.upper
    )


@pdf.dispatch
def _pdf(d: Uniform, x: PyTreeVar):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda l, u: _uniform_pdf(x, l, u), _tree.lower, _tree.upper)


@logpdf.dispatch
def _logpdf(d: Uniform, x: PyTreeVar):
    _tree = d.broadcast_params()
    return jtu.tree_map(
        lambda l, u: _uniform_log_pdf(x, l, u), _tree.lower, _tree.upper
    )


@logcdf.dispatch
def _logcdf(d: Uniform, x: PyTreeVar):
    _tree = d.broadcast_params()
    return jtu.tree_map(
        lambda l, u: _uniform_log_cdf(x, l, u), _tree.lower, _tree.upper
    )


@cdf.dispatch
def _cdf(d: Uniform, x: PyTreeVar):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda l, u: _uniform_cdf(x, l, u), _tree.lower, _tree.upper)


@mgf.dispatch
def _mgf(d: Uniform, t: PyTreeVar):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda l, u: _uniform_mgf(t, l, u), _tree.lower, _tree.upper)


@cf.dispatch
def _cf(d: Uniform, t: PyTreeVar):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda l, u: _uniform_cf(t, l, u), _tree.lower, _tree.upper)


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


def _uniform_log_cdf(x, lower, upper):
    return jtu.tree_map(lambda xx: jnp.log(xx - lower) - jnp.log(upper - lower), x)
