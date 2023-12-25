import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Float

from ..core import AbstractDistribution, PyTreeVar
from ..core._abstract_impls import _logpdf_impl, _pdf_impl, _truncate_impl
from ._base import (
    ContinuousUnivariateDistribution,
    DiscreteUnivariateDistribution,
    TransformedDistribution,
    UnivariateDistribution,
)


class TruncatedDistribution(TransformedDistribution):
    lower: Float
    upper: Float
    untruncated: AbstractDistribution

    def __init__(self, lower, upper, d):
        self.lower = lower
        self.upper = upper
        self.untruncated = d


class ContinuousTruncatedDistribution(TruncatedDistribution):
    def __init__(self, lower, upper, d):
        super().__init__(lower, upper, d)
        assert isinstance(d, ContinuousUnivariateDistribution)


class DiscreteTruncatedDistribution(TruncatedDistribution):
    def __init__(self, lower, upper, d):
        super().__init__(lower, upper, d)
        assert isinstance(d, DiscreteUnivariateDistribution)


@_truncate_impl.dispatch
def _truncate(d: ContinuousUnivariateDistribution, lower, upper):
    return ContinuousTruncatedDistribution(d, lower, upper)


@_truncate_impl.dispatch
def _truncate(d: DiscreteUnivariateDistribution, lower, upper):
    return DiscreteTruncatedDistribution(d, lower, upper)


@_pdf_impl.dispatch
def _pdf(d: TruncatedDistribution, x):
    return d.d.pdf(x) / (d.d.cdf(d.upper) - d.d.cdf(d.lower))
