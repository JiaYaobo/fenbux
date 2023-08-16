import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

from ...core import (
    _cdf_impl,
    _cf_impl,
    _check_params_equal_tree_strcutre,
    _entropy_impl,
    _intialize_params_tree,
    _kurtosis_impl,
    _logcdf_impl,
    _logpdf_impl,
    _mean_impl,
    _mgf_impl,
    _params_impl,
    _pdf_impl,
    _quantile_impl,
    _sf_impl,
    _skewness_impl,
    _standard_dev_impl,
    _support_impl,
    _variance_impl,
    KeyArray,
    PyTreeVar,
    rand,
    Shape,
)
from ...dist_special.uniform import (
    uniform_cdf,
    uniform_cf,
    uniform_logcdf,
    uniform_logpdf,
    uniform_mgf,
    uniform_pdf,
    uniform_ppf,
    uniform_sf,
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
        _check_params_equal_tree_strcutre(lower, upper, use_batch=use_batch)
        self.lower, self.upper = _intialize_params_tree(
            lower, upper, use_batch=use_batch, dtype=dtype
        )


@_params_impl.dispatch
def _params(d: Uniform):
    return (d.lower, d.upper)


@_support_impl.dispatch
def _domain(d: Uniform):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda l, u: (l, u), _tree.lower, _tree.upper)


@_mean_impl.dispatch
def _mean(d: Uniform):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda l, u: (l + u) / 2, _tree.lower, _tree.upper)


@_variance_impl.dispatch
def _variance(d: Uniform):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda l, u: (u - l) ** 2 / 12, _tree.lower, _tree.upper)


@_standard_dev_impl.dispatch
def _standard_dev(d: Uniform):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda l, u: (u - l) / jnp.sqrt(12), _tree.lower, _tree.upper)


@_kurtosis_impl.dispatch
def _kurtosis(d: Uniform):
    shape = d.broadcast_shapes()
    return full_pytree(shape, -6 / 5)


@_skewness_impl.dispatch
def _skewness(d: Uniform):
    shape = d.broadcast_shapes()
    return full_pytree(shape, 0.0)


@_entropy_impl.dispatch
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


@_quantile_impl.dispatch
def _quantile(d: Uniform, x: PyTreeVar):
    _tree = d.broadcast_params()
    return jtu.tree_map(
        lambda l, u: _uniform_quantile(x, l, u), _tree.lower, _tree.upper
    )


@_pdf_impl.dispatch
def _pdf(d: Uniform, x: PyTreeVar):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda l, u: _uniform_pdf(x, l, u), _tree.lower, _tree.upper)


@_logpdf_impl.dispatch
def _logpdf(d: Uniform, x: PyTreeVar):
    _tree = d.broadcast_params()
    return jtu.tree_map(
        lambda l, u: _uniform_log_pdf(x, l, u), _tree.lower, _tree.upper
    )


@_logcdf_impl.dispatch
def _logcdf(d: Uniform, x: PyTreeVar):
    _tree = d.broadcast_params()
    return jtu.tree_map(
        lambda l, u: _uniform_log_cdf(x, l, u), _tree.lower, _tree.upper
    )


@_cdf_impl.dispatch
def _cdf(d: Uniform, x: PyTreeVar):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda l, u: _uniform_cdf(x, l, u), _tree.lower, _tree.upper)


@_mgf_impl.dispatch
def _mgf(d: Uniform, t: PyTreeVar):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda l, u: _uniform_mgf(t, l, u), _tree.lower, _tree.upper)


@_cf_impl.dispatch
def _cf(d: Uniform, t: PyTreeVar):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda l, u: _uniform_cf(t, l, u), _tree.lower, _tree.upper)


@_sf_impl.dispatch
def _sf(d: Uniform, x: PyTreeVar):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda l, u: _uniform_sf(x, l, u), _tree.lower, _tree.upper)


def _uniform_log_pdf(x, lower, upper):
    return jtu.tree_map(lambda xx: uniform_logpdf(xx, lower, upper), x)


def _uniform_pdf(x, lower, upper):
    return jtu.tree_map(lambda xx: uniform_pdf(xx, lower, upper), x)


def _uniform_log_cdf(x, lower, upper):
    return jtu.tree_map(lambda xx: uniform_logcdf(xx, lower, upper), x)


def _uniform_cdf(x, lower, upper):
    return jtu.tree_map(lambda xx: uniform_cdf(xx, lower, upper), x)


def _uniform_quantile(x, lower, upper):
    return jtu.tree_map(lambda xx: uniform_ppf(xx, lower, upper), x)


def _uniform_mgf(t, lower, upper):
    return jtu.tree_map(lambda tt: uniform_mgf(tt, lower, upper), t)


def _uniform_cf(t, lower, upper):
    return jtu.tree_map(
        lambda tt: jnp.exp(1j * tt * upper) - jnp.exp(1j * tt * lower), t
    )


def _uniform_sf(x, lower, upper):
    return jtu.tree_map(lambda xx: uniform_sf(xx, lower, upper), x)
