from typing import Tuple, Union

from jaxtyping import PyTree
from plum import Dispatcher

from ._dist import AbstractDistribution
from ._typing import DTypeLikeFloat, DTypeLikeInt, KeyArray, Shape


_fenbux_dispatch = Dispatcher()


@_fenbux_dispatch.abstract
def _params_impl(dist: AbstractDistribution) -> Tuple[PyTree, ...]:
    ...


@_fenbux_dispatch.abstract
def _support_impl(dist: AbstractDistribution) -> Tuple[PyTree, ...]:
    ...


@_fenbux_dispatch.abstract
def _mean_impl(dist: AbstractDistribution) -> PyTree:
    ...


@_fenbux_dispatch.abstract
def _variance_impl(dist: AbstractDistribution) -> PyTree:
    ...


@_fenbux_dispatch.abstract
def _standard_dev_impl(dist: AbstractDistribution) -> PyTree:
    ...


@_fenbux_dispatch.abstract
def _skewness_impl(dist: AbstractDistribution) -> PyTree:
    ...


@_fenbux_dispatch.abstract
def _kurtosis_impl(dist: AbstractDistribution) -> PyTree:
    ...


@_fenbux_dispatch.abstract
def _entropy_impl(dist: AbstractDistribution) -> PyTree:
    ...


@_fenbux_dispatch.abstract
def _pdf_impl(dist: AbstractDistribution, x: PyTree) -> PyTree:
    ...


@_fenbux_dispatch.abstract
def _logpdf_impl(dist: AbstractDistribution, x: PyTree) -> PyTree:
    ...


@_fenbux_dispatch.abstract
def _logcdf_impl(dist: AbstractDistribution, x: PyTree) -> PyTree:
    ...


@_fenbux_dispatch.abstract
def _cdf_impl(dist: AbstractDistribution, x: PyTree) -> PyTree:
    ...


@_fenbux_dispatch.abstract
def _rand_impl(
    dist: AbstractDistribution,
    key: KeyArray,
    shape: Shape,
    dtype: Union[DTypeLikeFloat, DTypeLikeInt],
) -> PyTree:
    ...


@_fenbux_dispatch.abstract
def _quantile_impl(dist: AbstractDistribution, p: PyTree) -> PyTree:
    ...


@_fenbux_dispatch.abstract
def _logpmf_impl(dist: AbstractDistribution, x: PyTree) -> PyTree:
    ...


@_fenbux_dispatch.abstract
def _pmf_impl(dist: AbstractDistribution, x: PyTree) -> PyTree:
    ...


@_fenbux_dispatch.abstract
def _mgf_impl(dist: AbstractDistribution, t: PyTree) -> PyTree:
    ...


@_fenbux_dispatch.abstract
def _sf_impl(dist: AbstractDistribution, x: PyTree) -> PyTree:
    ...


@_fenbux_dispatch.abstract
def _cf_impl(dist: AbstractDistribution, t: PyTree) -> PyTree:
    ...


@_fenbux_dispatch.abstract
def _affine_impl(d: AbstractDistribution, loc: PyTree, scale: PyTree) -> PyTree:
    ...
