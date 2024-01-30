import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import ArrayLike

from ...core import (
    _cdf_impl,
    _cf_impl,
    _check_params_equal_tree_strcutre,
    _intialize_params_tree,
    _kurtosis_impl,
    _logcdf_impl,
    _logpmf_impl,
    _mean_impl,
    _mgf_impl,
    _params_impl,
    _pmf_impl,
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
from ...dist_math.binomial import (
    binom_cdf,
    binom_cf,
    binom_logcdf,
    binom_logpmf,
    binom_mgf,
    binom_pmf,
    binom_ppf,
    binom_sf,
)
from ...random_utils import split_tree
from ...tree_utils import tree_map_dist_at
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
        >>> from fenbux import logpmf
        >>> from fenbux.univariate import Binomial
        >>> dist = Binomial(10, 0.5)
        >>> logpmf(dist, jnp.ones((10, )))
    """

    n: PyTreeVar
    p: PyTreeVar

    def __init__(self, n=0.0, p=0.0, dtype=jnp.float_, use_batch=False):
        _check_params_equal_tree_strcutre(n, p, use_batch=use_batch)
        self.n, self.p = _intialize_params_tree(n, p, use_batch=use_batch, dtype=dtype)


@_params_impl.dispatch
def _params(d: Binomial):
    return (d.n, d.p)


@_support_impl.dispatch
def _support(d: Binomial):
    d = d.broadcast_params()
    return jtu.tree_map(lambda n: jnp.zeros_like(n), d.n), jtu.tree_map(
        lambda n: n, d.n
    )


@_mean_impl.dispatch
def _mean(d: Binomial):
    d = d.broadcast_params()
    return jtu.tree_map(lambda p, n: p * n, d.p, d.n)


@_variance_impl.dispatch
def _variance(d: Binomial):
    d = d.broadcast_params()
    return jtu.tree_map(lambda p, n: p * (1 - p) * n, d.p, d.n)


@_standard_dev_impl.dispatch
def _standard_dev(d: Binomial):
    d = d.broadcast_params()
    return jtu.tree_map(lambda p, n: jnp.sqrt(p * (1 - p) * n), d.p, d.n)


@_skewness_impl.dispatch
def _skewness(d: Binomial):
    d = d.broadcast_params()
    return jtu.tree_map(lambda p, n: (1 - 2 * p) / jnp.sqrt(p * (1 - p) * n), d.p, d.n)


@_kurtosis_impl.dispatch
def _kurtosis(d: Binomial):
    d = d.broadcast_params()
    return jtu.tree_map(
        lambda p, n: (1 - 6 * p * (1 - p)) / (p * (1 - p) * n), d.p, d.n
    )


@_logpmf_impl.dispatch
def _logpmf(d: Binomial, x: ArrayLike):
    d = d.broadcast_params()
    return tree_map_dist_at(binom_logpmf, d, x)


@_pmf_impl.dispatch
def _pmf(d: Binomial, x: ArrayLike):
    d = d.broadcast_params()
    return tree_map_dist_at(binom_pmf, d, x)


@_logcdf_impl.dispatch
def _logcdf(d: Binomial, x: ArrayLike):
    d = d.broadcast_params()
    return tree_map_dist_at(binom_logcdf, d, x)


@_cdf_impl.dispatch
def _cdf(d: Binomial, x: ArrayLike):
    d = d.broadcast_params()
    return tree_map_dist_at(binom_cdf, d, x)


@_quantile_impl.dispatch
def _quantile(d: Binomial, q: ArrayLike):
    d = d.broadcast_params()
    return tree_map_dist_at(binom_ppf, d, q)


@_mgf_impl.dispatch
def _mgf(d: Binomial, t: ArrayLike):
    d = d.broadcast_params()
    return tree_map_dist_at(binom_mgf, d, t)


@_cf_impl.dispatch
def _cf(d: Binomial, t: ArrayLike):
    d = d.broadcast_params()
    return tree_map_dist_at(binom_cf, d, t)


@_sf_impl.dispatch
def _sf(d: Binomial, x: ArrayLike):
    d = d.broadcast_params()
    return tree_map_dist_at(binom_sf, d, x)


@_rand_impl.dispatch
def _rand(d: Binomial, key: KeyArray, shape: Shape = (), dtype: DTypeLikeFloat = float):
    _dist = d.broadcast_params()
    _key_tree = split_tree(key, _dist.n)
    rvs = jtu.tree_map(
        lambda p, n, k: jr.binomial(k, n, p, shape=shape, dtype=dtype),
        _dist.p,
        _dist.n,
        _key_tree,
    )
    return rvs
