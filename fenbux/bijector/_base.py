import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import ArrayLike

from ..core import (
    _logcdf_impl,
    _logpdf_impl,
    _logsf_impl,
    _rand_impl,
    AbstractDistribution,
)
from ..univariate import UnivariateDistribution
from ._abstract_impls import evaluate, ildj, inverse, is_increasing, transform
from ._typing import Bijector


class AbstractBijectorTransformedDistribution(eqx.Module):
    dist: AbstractDistribution
    bijector: Bijector

    def __init__(self, dist: AbstractDistribution, bij: Bijector):
        self.dist = dist
        self.bijector = bij


class UnivariateBijectorTransformedDistribution(
    AbstractBijectorTransformedDistribution
):
    def __init__(self, dist: UnivariateDistribution, bij: Bijector):
        super().__init__(dist, bij)


@transform.dispatch
def _transform(
    d: AbstractDistribution, bij: Bijector
) -> AbstractBijectorTransformedDistribution:
    return AbstractBijectorTransformedDistribution(d, bij)


@transform.dispatch
def _transform(
    d: UnivariateDistribution, bij: Bijector
) -> UnivariateBijectorTransformedDistribution:
    return UnivariateBijectorTransformedDistribution(d, bij)


@_logcdf_impl.dispatch
def _logcdf(d: UnivariateBijectorTransformedDistribution, x: ArrayLike):
    y = evaluate(inverse(d.bijector), x)
    _lcdf = _logcdf_impl(d.dist, y)
    _lsf = _logsf_impl(d.dist, y)
    _is_inc = is_increasing(d.bijector)
    return jtu.tree_map(
        lambda __lcdf, __lsf: jnp.where(_is_inc, __lcdf, __lsf), _lcdf, _lsf
    )


@_logpdf_impl.dispatch
def _logpdf(d: UnivariateBijectorTransformedDistribution, x: ArrayLike):
    y = evaluate(inverse(d.bijector), x)
    _ildj = ildj(d.bijector, x)
    _lp = _logpdf_impl(d.dist, y)
    return jtu.tree_map(lambda __lp: __lp + _ildj, _lp)


@_rand_impl.dispatch
def _rand(d: UnivariateBijectorTransformedDistribution, key, shape, dtype):
    rvs = _rand_impl(d.dist, key, shape, dtype)
    return jtu.tree_map(lambda rv: evaluate(d.bijector, rv), rvs)
