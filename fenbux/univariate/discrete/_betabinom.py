import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import ArrayLike

from ...core import (
    _check_params_equal_tree_strcutre,
    _intialize_params_tree,
    _kurtosis_impl,
    _logpmf_impl,
    _mean_impl,
    _params_impl,
    _pmf_impl,
    _rand_impl,
    _skewness_impl,
    _standard_dev_impl,
    _support_impl,
    _variance_impl,
    DTypeLikeFloat,
    KeyArray,
    PyTreeVar,
    Shape,
)
from ...dist_math.betabinom import (
    betabinom_logpmf,
    betabinom_pmf,
)
from ...extension._jax_random_ext import betabinom
from ...random_utils import split_tree
from ...tree_utils import tree_map_dist_at
from .._base import DiscreteUnivariateDistribution


class BetaBinomial(DiscreteUnivariateDistribution):
    """BetaBinomial distribution.

            X ~ BetaBinomial(n, a, b)

    Args:
        n (PyTree): Number of trials.
        a (PyTree): Shape parameter a.
        b (PyTree): Shape parameter b.
        dtype (jax.numpy.dtype): dtype of the distribution, default jnp.float_.
        use_batch (bool): Whether to use with vmap. Default False.

    Examples:
        >>> import jax.numpy as jnp
        >>> from fenbux import logpmf
        >>> from fenbux.univariate import BetaBinomial
        >>> dist = BetaBinomial(10, 1.0, 1.0)
        >>> logpmf(dist, jnp.ones((10, )))
    """

    n: PyTreeVar
    a: PyTreeVar
    b: PyTreeVar

    def __init__(self, n=1, a=1.0, b=1.0, dtype=jnp.float_, use_batch=False):
        _check_params_equal_tree_strcutre(n, a, b, use_batch=use_batch)
        self.n, self.a, self.b = _intialize_params_tree(
            n, a, b, use_batch=use_batch, dtype=dtype
        )


@_params_impl.dispatch
def _params_impl(
    dist: BetaBinomial,
):
    return (dist.n, dist.a, dist.b)


@_support_impl.dispatch
def _support_impl(
    dist: BetaBinomial,
):
    dist = dist.broadcast_params()
    return jtu.tree_map(lambda _n: jnp.full_like(_n, 0.0), dist.n), jtu.tree_map(
        lambda _n: _n, dist.n
    )


@_mean_impl.dispatch
def _mean_impl(
    dist: BetaBinomial,
):
    dist = dist.broadcast_params()
    return jtu.tree_map(
        lambda a, b, n: n * a / (a + b),
        dist.a,
        dist.b,
        dist.n,
    )


@_variance_impl.dispatch
def _variance_impl(
    dist: BetaBinomial,
):
    dist = dist.broadcast_params()
    return jtu.tree_map(
        lambda a, b, n: n * a * b * (a + b + n) / ((a + b) ** 2 * (a + b + 1)),
        dist.a,
        dist.b,
        dist.n,
    )


@_standard_dev_impl.dispatch
def _standard_dev_impl(
    dist: BetaBinomial,
):
    dist = dist.broadcast_params()
    return jtu.tree_map(
        lambda a, b, n: jnp.sqrt(
            n * a * b * (a + b + n) / ((a + b) ** 2 * (a + b + 1))
        ),
        dist.a,
        dist.b,
        dist.n,
    )


@_skewness_impl.dispatch
def _skewness_impl(
    dist: BetaBinomial,
):
    dist = dist.broadcast_params()
    return jtu.tree_map(
        lambda a, b, n: 2
        * (b - a)
        * jnp.sqrt(a + b + 1)
        / ((a + b + 2) * jnp.sqrt(n * a * b * (a + b + n))),
        dist.a,
        dist.b,
        dist.n,
    )


@_kurtosis_impl.dispatch
def _kurtosis_impl(
    dist: BetaBinomial,
):
    dist = dist.broadcast_params()
    return jtu.tree_map(
        lambda a, b, n: (
            (
                (
                    (a + b) * (a + b - 1 + 6 * n)
                    + 3 * a * b * (n - 2)
                    + 6 * n**2
                    - 3 * (a / (a + b)) * b * n * (6 - n)
                    - 18 * (a / (a + b)) * (b / (a + b)) * n**2
                )
                * (a + b) ** 2
                * (1 + a + b)
            )
            / (n * a * b * (a + b + 2) * (a + b + 3) * (a + b + n))
            - 3
        ),
        dist.a,
        dist.b,
        dist.n,
    )


@_logpmf_impl.dispatch
def _logpmf_impl(
    dist: BetaBinomial,
    x: ArrayLike,
):
    dist = dist.broadcast_params()
    return tree_map_dist_at(betabinom_logpmf, dist, x)


@_pmf_impl.dispatch
def _pmf_impl(
    dist: BetaBinomial,
    x: ArrayLike,
):
    dist = dist.broadcast_params()
    return tree_map_dist_at(betabinom_pmf, dist, x)


@_rand_impl.dispatch
def _rand_impl(
    dist: BetaBinomial, key: KeyArray, shape: Shape = (), dtype: DTypeLikeFloat = float
):
    _dist = dist.broadcast_params()
    _key_tree = split_tree(key, _dist.n)
    rvs = jtu.tree_map(
        lambda n, a, b, k: betabinom(k, n, a, b, shape=shape, dtype=dtype),
        _dist.a,
        _dist.b,
        _dist.n,
        _key_tree,
    )
    return rvs
