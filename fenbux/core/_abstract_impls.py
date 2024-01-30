from typing import Tuple

from fbx_plum import Dispatcher
from jaxtyping import ArrayLike, PyTree

from ._dist import AbstractDistribution
from ._typing import DTypeLike, KeyArray, Shape


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
def _pdf_impl(dist: AbstractDistribution, x: ArrayLike) -> PyTree:
    ...


@_fenbux_dispatch.abstract
def _logpdf_impl(dist: AbstractDistribution, x: ArrayLike) -> PyTree:
    ...


@_fenbux_dispatch.abstract
def _logcdf_impl(dist: AbstractDistribution, x: ArrayLike) -> PyTree:
    ...


@_fenbux_dispatch.abstract
def _cdf_impl(dist: AbstractDistribution, x: ArrayLike) -> PyTree:
    ...


@_fenbux_dispatch.abstract
def _rand_impl(
    dist: AbstractDistribution,
    key: KeyArray,
    shape: Shape,
    dtype: DTypeLike,
) -> PyTree:
    ...


@_fenbux_dispatch.abstract
def _quantile_impl(dist: AbstractDistribution, p: ArrayLike) -> PyTree:
    ...


@_fenbux_dispatch.abstract
def _logpmf_impl(dist: AbstractDistribution, x: ArrayLike) -> PyTree:
    ...


@_fenbux_dispatch.abstract
def _pmf_impl(dist: AbstractDistribution, x: ArrayLike) -> PyTree:
    ...


@_fenbux_dispatch.abstract
def _mgf_impl(dist: AbstractDistribution, t: ArrayLike) -> PyTree:
    ...


@_fenbux_dispatch.abstract
def _sf_impl(dist: AbstractDistribution, x: ArrayLike) -> PyTree:
    ...


@_fenbux_dispatch.abstract
def _cf_impl(dist: AbstractDistribution, t: ArrayLike) -> PyTree:
    ...


@_fenbux_dispatch.abstract
def _affine_impl(d: AbstractDistribution, loc: ArrayLike, scale: ArrayLike) -> PyTree:
    ...


@_fenbux_dispatch.abstract
def _truncate_impl(
    d: AbstractDistribution, lower: ArrayLike, upper: ArrayLike
) -> AbstractDistribution:
    ...


@_fenbux_dispatch.abstract
def _censor_impl(
    d: AbstractDistribution, lower: ArrayLike, upper: ArrayLike
) -> AbstractDistribution:
    ...
