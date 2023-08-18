import equinox as eqx
import jax.tree_util as jtu

from ..core import (
    _logpdf_impl,
    AbstractDistribution,
    _rand_impl,
)
from ..univariate import UnivariateDistribution
from ._abstract_impls import bijector, transform, transformed, value_and_ladj
from ._typing import Bijector


class AbstractTransformedDistribution(eqx.Module):
    dist: AbstractDistribution
    transform: Bijector

    def __init__(self, dist: AbstractDistribution, bij: Bijector):
        self.dist = dist
        self.transform = bij


class UnivariateTransformedDistribution(AbstractTransformedDistribution):
    def __init__(self, dist: UnivariateDistribution, bij: Bijector):
        super().__init__(dist, bij)


@transformed.dispatch
def _transformed(
    d: AbstractDistribution, bij: Bijector
) -> AbstractTransformedDistribution:
    return AbstractTransformedDistribution(d, bij)


@transformed.dispatch
def _transformed(d: AbstractDistribution) -> AbstractTransformedDistribution:
    return AbstractTransformedDistribution(d, bijector(d))


@transformed.dispatch
def _transformed(
    d: UnivariateDistribution, bij: Bijector
) -> UnivariateTransformedDistribution:
    return UnivariateTransformedDistribution(d, bij)


@_logpdf_impl.dispatch
def _logpdf(d: UnivariateTransformedDistribution, x):
    val, ladj = value_and_ladj(d.transform, x)
    return jtu.tree_map(lambda v, j: _logpdf_impl(d.dist, v) + j, val, ladj)


@_rand_impl.dispatch
def _rand(d: UnivariateTransformedDistribution, key, shape, dtype):
    rvs = _rand_impl(d.dist, key, shape, dtype)
    return jtu.tree_map(lambda rv: transform(d.transform, rv), rvs)
