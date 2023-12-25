import jax.tree_util as jtu
from jaxtyping import ArrayLike, Int, PyTree

from ..core import AbstractDistribution, logcdf, logpdf, logpmf, params, rand
from ..core._abstract_impls import (
    _logcdf_impl,
    _logpdf_impl,
    _logpmf_impl,
    _params_impl,
    _rand_impl,
)
from ..core._typing import DTypeLikeFloat, KeyArray, Shape


class IndependentIdenticalDistribution(AbstractDistribution):
    dists: PyTree
    dist_params: PyTree
    dist_type: AbstractDistribution

    """A distribution that is a collection of independent identical distributions.
    
    Args:
        dist (AbstractDistribution): A distribution object.
        n (int): Number of independent identical distributions.
        tree_struct (PyTree): A tree structure of the distribution.
        
    Example:
        >>> from fenbux import Normal, IndependentIdenticalDistribution
        >>> dist = Normal(0.0, 1.0)
        >>> iid_dist = IndependentIdenticalDistribution(dist, n=10)
    """

    def __init__(self, dist: AbstractDistribution, n: Int = None, tree_struct=None):
        self.dist_params = params(dist)
        self.dist_type = type(dist)
        if tree_struct is None and n is None:
            raise ValueError("Either n or tree_struct must be provided.")

        if n is not None and tree_struct is None:
            self.dists = [type(dist)(*self.dist_params) for _ in range(n)]

        if n is None and tree_struct is not None:
            self.dists = jtu.tree_map(
                lambda _: type(dist)(*self.dist_params), tree_struct
            )

        if n is not None and tree_struct is not None:
            self.dists = jtu.tree_map(
                lambda _: [type(dist)(*self.dist_params) for _ in range(n)], tree_struct
            )


@_params_impl.dispatch
def _params(d: IndependentIdenticalDistribution):
    return jtu.tree_map(
        lambda dist: params(dist),
        d.dists,
        is_leaf=lambda x: isinstance(x, d.dist_type),
    )


@_logpdf_impl.dispatch
def _log_pdf(d: IndependentIdenticalDistribution, x: ArrayLike):
    return jtu.tree_map(
        lambda dd: logpdf(dd, x),
        d.dists,
        is_leaf=lambda x: isinstance(x, d.dist_type),
    )


@_logcdf_impl.dispatch
def _logcdf(d: IndependentIdenticalDistribution, x: ArrayLike):
    return jtu.tree_map(
        lambda dd: logcdf(dd, x),
        d.dists,
        is_leaf=lambda x: isinstance(x, d.dist_type),
    )


@_logpmf_impl.dispatch
def _logpmf(d: IndependentIdenticalDistribution, x: ArrayLike):
    return jtu.tree_map(
        lambda dd: logpmf(dd, x),
        d.dists,
        is_leaf=lambda x: isinstance(x, d.dist_type),
    )


@_rand_impl.dispatch
def _rand(
    d: IndependentIdenticalDistribution,
    key: KeyArray,
    shape: Shape = (),
    dtype: DTypeLikeFloat = float,
):
    return jtu.tree_map(
        lambda dist: rand(dist, key, shape, dtype),
        d.dists,
        is_leaf=lambda x: not isinstance(x, d.dist_type),
    )
