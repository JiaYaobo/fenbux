import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import ArrayLike

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
    _rand_impl,
    _sf_impl,
    _skewness_impl,
    _standard_dev_impl,
    _support_impl,
    _variance_impl,
    DTypeLikeFloat,
    KeyArray,
    PyTreeVar,
    Shape,
)
from ...dist_math.uniform import (
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
from ...tree_utils import full_pytree, tree_map_dist_at
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
        >>> from fenbux import logpdf
        >>> from fenbux.univariate import Uniform
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
    dist = d.broadcast_params()
    return dist.lower, dist.upper


@_mean_impl.dispatch
def _mean(d: Uniform):
    dist = d.broadcast_params()
    return jtu.tree_map(lambda l, u: (l + u) / 2, dist.lower, dist.upper)


@_variance_impl.dispatch
def _variance(d: Uniform):
    dist = d.broadcast_params()
    return jtu.tree_map(lambda l, u: (u - l) ** 2 / 12, dist.lower, dist.upper)


@_standard_dev_impl.dispatch
def _standard_dev(d: Uniform):
    dist = d.broadcast_params()
    return jtu.tree_map(lambda l, u: (u - l) / jnp.sqrt(12), dist.lower, dist.upper)


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
    dist = d.broadcast_params()
    return jtu.tree_map(lambda l, u: jnp.log(u - l), dist.lower, dist.upper)


@_rand_impl.dispatch
def _rand(d: Uniform, key: KeyArray, shape: Shape = (), dtype: DTypeLikeFloat = float):
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
def _quantile(d: Uniform, x: ArrayLike):
    _tree = d.broadcast_params()
    return tree_map_dist_at(uniform_ppf, _tree, x)


@_pdf_impl.dispatch
def _pdf(d: Uniform, x: ArrayLike):
    _tree = d.broadcast_params()
    return tree_map_dist_at(uniform_pdf, _tree, x)


@_logpdf_impl.dispatch
def _logpdf(d: Uniform, x: ArrayLike):
    _tree = d.broadcast_params()
    return tree_map_dist_at(uniform_logpdf, _tree, x)


@_logcdf_impl.dispatch
def _logcdf(d: Uniform, x: ArrayLike):
    _tree = d.broadcast_params()
    return tree_map_dist_at(uniform_logcdf, _tree, x)


@_cdf_impl.dispatch
def _cdf(d: Uniform, x: ArrayLike):
    _tree = d.broadcast_params()
    return tree_map_dist_at(uniform_cdf, _tree, x)


@_mgf_impl.dispatch
def _mgf(d: Uniform, t: ArrayLike):
    _tree = d.broadcast_params()
    return tree_map_dist_at(uniform_mgf, _tree, t)


@_cf_impl.dispatch
def _cf(d: Uniform, t: ArrayLike):
    _tree = d.broadcast_params()
    return tree_map_dist_at(uniform_cf, _tree, t)


@_sf_impl.dispatch
def _sf(d: Uniform, x: ArrayLike):
    _tree = d.broadcast_params()
    return tree_map_dist_at(uniform_sf, _tree, x)
