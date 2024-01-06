import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import ArrayLike

from ..core import AbstractDistribution
from ..core._abstract_impls import (
    _cdf_impl,
    _censor_impl,
    _logcdf_impl,
    _logpdf_impl,
    _logpmf_impl,
    _pdf_impl,
    _pmf_impl,
)
from ._base import (
    ContinuousUnivariateDistribution,
    DiscreteUnivariateDistribution,
    TransformedDistribution,
)


class CensoredDistribution(TransformedDistribution):
    lower: ArrayLike
    upper: ArrayLike
    uncensored: AbstractDistribution

    def __init__(self, lower, upper, d):
        self.lower = lower
        self.upper = upper
        self.uncensored = d


class ContinuousCensoredDistribution(CensoredDistribution):
    def __init__(self, lower, upper, d):
        super().__init__(lower, upper, d)


class DiscreteCensoredDistribution(CensoredDistribution):
    def __init__(self, lower, upper, d):
        super().__init__(lower, upper, d)


@_censor_impl.dispatch
def _censor(d: ContinuousUnivariateDistribution, lower, upper):
    return ContinuousCensoredDistribution(lower, upper, d)


@_censor_impl.dispatch
def _censor(d: DiscreteUnivariateDistribution, lower, upper):
    return DiscreteCensoredDistribution(lower, upper, d)


@_pdf_impl.dispatch
def _pdf(d: ContinuousCensoredDistribution, x: ArrayLike):
    px = _pdf_impl(d.uncensored, x)
    px = jtu.tree_map(lambda _px: jnp.where(x < d.lower, 0.0, _px), px)
    px = jtu.tree_map(lambda _px: jnp.where(x > d.upper, 0.0, _px), px)
    return px


@_logpdf_impl.dispatch
def _logpdf(d: ContinuousCensoredDistribution, x: ArrayLike):
    px = _logpdf_impl(d.uncensored, x)
    px = jtu.tree_map(lambda _px: jnp.where(x < d.lower, -jnp.inf, _px), px)
    px = jtu.tree_map(lambda _px: jnp.where(x > d.upper, -jnp.inf, _px), px)
    return px


@_cdf_impl.dispatch
def _cdf(d: ContinuousCensoredDistribution, x: ArrayLike):
    px = _cdf_impl(d.uncensored, x)
    px = jtu.tree_map(lambda _px: jnp.where(x < d.lower, 0.0, _px), px)
    px = jtu.tree_map(lambda _px: jnp.where(x > d.upper, 1.0, _px), px)
    return px


@_logcdf_impl.dispatch
def _logcdf(d: ContinuousCensoredDistribution, x: ArrayLike):
    px = _logcdf_impl(d.uncensored, x)
    px = jtu.tree_map(lambda _px: jnp.where(x < d.lower, -jnp.inf, _px), px)
    px = jtu.tree_map(lambda _px: jnp.where(x > d.upper, 0.0, _px), px)
    return px


@_pmf_impl.dispatch
def _pmf(d: DiscreteCensoredDistribution, x: ArrayLike):
    px = _pmf_impl(d.uncensored, x)
    px = jtu.tree_map(lambda _px: jnp.where(x < d.lower, 0.0, _px), px)
    px = jtu.tree_map(lambda _px: jnp.where(x > d.upper, 0.0, _px), px)
    return px


@_logpmf_impl.dispatch
def _logpmf(d: DiscreteCensoredDistribution, x: ArrayLike):
    px = _logpmf_impl(d.uncensored, x)
    px = jtu.tree_map(lambda _px: jnp.where(x < d.lower, -jnp.inf, _px), px)
    px = jtu.tree_map(lambda _px: jnp.where(x > d.upper, -jnp.inf, _px), px)
    return px
