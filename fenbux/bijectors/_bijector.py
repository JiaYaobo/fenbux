### Map support(P) to R^d
from ..univariate import (
    ContinuousUnivariateDistribution,
    DiscreteUnivariateDistribution,
)
from ..univariate.continuous import Normal
from ._base import Identity
from ._func import bijector


@bijector.dispatch
def _bijector(d: Normal):
    return Identity()

@bijector.dispatch
def _bijector(d: DiscreteUnivariateDistribution):
    return Identity()