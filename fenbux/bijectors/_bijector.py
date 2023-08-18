### Map support(P) to R^d
from ..univariate import (
    ContinuousUnivariateDistribution,
    DiscreteUnivariateDistribution,
)
from ..univariate.continuous import Normal
from ._abstract_impls import bijector
from ._base import Identity


@bijector.dispatch
def _bijector(d: Normal):
    return Identity()

@bijector.dispatch
def _bijector(d: DiscreteUnivariateDistribution):
    return Identity()