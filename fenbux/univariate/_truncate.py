import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import ArrayLike

from ..core import AbstractDistribution
from ..core._abstract_impls import (
    _cdf_impl,
    _logcdf_impl,
    _logpdf_impl,
    _pdf_impl,
    _pmf_impl,
    _support_impl,
    _truncate_impl,
)
from ._base import (
    ContinuousUnivariateDistribution,
    DiscreteUnivariateDistribution,
    TransformedDistribution,
)


class TruncatedDistribution(TransformedDistribution):
    lower: ArrayLike
    upper: ArrayLike
    untruncated: AbstractDistribution

    def __init__(self, lower, upper, d):
        self.lower = lower
        self.upper = upper
        self.untruncated = d


class ContinuousTruncatedDistribution(TruncatedDistribution):
    def __init__(self, lower, upper, d):
        super().__init__(lower, upper, d)


class DiscreteTruncatedDistribution(TruncatedDistribution):
    def __init__(self, lower, upper, d):
        super().__init__(lower, upper, d)


@_truncate_impl.dispatch
def _truncate(d: ContinuousUnivariateDistribution, lower, upper):
    if lower is None:
        lower = _support_impl(d)[0]
    if upper is None:
        upper = _support_impl(d)[1]
    return ContinuousTruncatedDistribution(lower, upper, d)


@_truncate_impl.dispatch
def _truncate(d: DiscreteUnivariateDistribution, lower, upper):
    return DiscreteTruncatedDistribution(lower, upper, d)


@_pdf_impl.dispatch
def _pdf(d: ContinuousTruncatedDistribution, x: ArrayLike):
    pdfs = _pdf_impl(d.untruncated, x)
    cdfs_lower = _cdf_impl(d.untruncated, d.lower)
    cdfs_upper = _cdf_impl(d.untruncated, d.upper)
    return jtu.tree_map(
        lambda _pdf, _cdf_lower, _cdf_upper: _pdf / (_cdf_upper - _cdf_lower),
        pdfs,
        cdfs_lower,
        cdfs_upper,
    )


@_pmf_impl.dispatch
def _pmf(d: DiscreteTruncatedDistribution, x: ArrayLike):
    pdfs = _pmf_impl(d.untruncated, x)
    cdfs_lower = _cdf_impl(d.untruncated, d.lower)
    cdfs_upper = _cdf_impl(d.untruncated, d.upper)
    return jtu.tree_map(
        lambda _pdf, _cdf_lower, _cdf_upper: _pdf / (_cdf_upper - _cdf_lower),
        pdfs,
        cdfs_lower,
        cdfs_upper,
    )


@_logpdf_impl.dispatch
def _logpdf(d: ContinuousTruncatedDistribution, x: ArrayLike):
    pdfs = _pdf_impl(d.untruncated, x)
    cdfs_lower = _cdf_impl(d.untruncated, d.lower)
    cdfs_upper = _cdf_impl(d.untruncated, d.upper)
    return jtu.tree_map(
        lambda _pdf, _cdf_lower, _cdf_upper: jnp.log(_pdf)
        - jnp.log(_cdf_upper - _cdf_lower),
        pdfs,
        cdfs_lower,
        cdfs_upper,
    )


@_logpdf_impl.dispatch
def _logpdf(d: DiscreteTruncatedDistribution, x: ArrayLike):
    pdfs = _pmf_impl(d.untruncated, x)
    cdfs_lower = _cdf_impl(d.untruncated, d.lower)
    cdfs_upper = _cdf_impl(d.untruncated, d.upper)
    return jtu.tree_map(
        lambda _pdf, _cdf_lower, _cdf_upper: jnp.log(_pdf)
        - jnp.log(_cdf_upper - _cdf_lower),
        pdfs,
        cdfs_lower,
        cdfs_upper,
    )


@_cdf_impl.dispatch
def _cdf(d: TruncatedDistribution, x: ArrayLike):
    cdfs = _cdf_impl(d.untruncated, x)
    cdfs_lower = _cdf_impl(d.untruncated, d.lower)
    cdfs_upper = _cdf_impl(d.untruncated, d.upper)
    return jtu.tree_map(
        lambda _cdf, _cdf_lower, _cdf_upper: (_cdf - _cdf_lower)
        / (_cdf_upper - _cdf_lower),
        cdfs,
        cdfs_lower,
        cdfs_upper,
    )


@_logcdf_impl.dispatch
def _logcdf(d: TruncatedDistribution, x: ArrayLike):
    cdfs = _cdf_impl(d.untruncated, x)
    cdfs_lower = _cdf_impl(d.untruncated, d.lower)
    cdfs_upper = _cdf_impl(d.untruncated, d.upper)
    return jtu.tree_map(
        lambda _cdf, _cdf_lower, _cdf_upper: jnp.log(_cdf - _cdf_lower)
        - jnp.log(_cdf_upper - _cdf_lower),
        cdfs,
        cdfs_lower,
        cdfs_upper,
    )
