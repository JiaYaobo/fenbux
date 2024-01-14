import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import ArrayLike

from ..core import AbstractDistribution, PyTreeVar
from ..core._abstract_impls import (
    _affine_impl,
    _cf_impl,
    _entropy_impl,
    _kurtosis_impl,
    _logpdf_impl,
    _mean_impl,
    _mgf_impl,
    _params_impl,
    _pdf_impl,
    _skewness_impl,
    _standard_dev_impl,
    _variance_impl,
)
from ._base import (
    ContinuousUnivariateDistribution,
    DiscreteUnivariateDistribution,
    TransformedDistribution,
)


class AffineDistribution(TransformedDistribution):
    loc: ArrayLike
    scale: ArrayLike
    dist: AbstractDistribution

    def __init__(self, loc, scale, d):
        self.loc = loc
        self.scale = scale
        self.dist = d


class ContinuousAffineDistribution(AffineDistribution):
    def __init__(self, loc, scale, d):
        super().__init__(loc, scale, d)
        assert isinstance(d, ContinuousUnivariateDistribution)


class DiscreteAffineDistribution(AffineDistribution):
    def __init__(self, loc, scale, d):
        super().__init__(loc, scale, d)
        assert isinstance(d, DiscreteUnivariateDistribution)


@_affine_impl.dispatch
def _affine_general(d: TransformedDistribution, loc, scale):
    return AffineDistribution(loc, scale, d)


@_affine_impl.dispatch
def _affine1(d: ContinuousUnivariateDistribution, loc, scale):
    return ContinuousAffineDistribution(loc, scale, d)


@_affine_impl.dispatch
def _affine2(d: DiscreteUnivariateDistribution, loc, scale):
    return DiscreteAffineDistribution(loc, scale, d)


@_params_impl.dispatch
def _params(d: AffineDistribution):
    return (d.loc, d.scale, _params_impl(d.dist))


@_mean_impl.dispatch
def _mean(d: AffineDistribution):
    _mean = _mean_impl(d.dist)
    return jtu.tree_map(lambda _m: _m * d.scale + d.loc, _mean)


@_variance_impl.dispatch
def _variance(d: AffineDistribution):
    _variance = _variance_impl(d.dist)
    return jtu.tree_map(lambda _v: _v * d.scale**2, _variance)


@_standard_dev_impl.dispatch
def _standard_dev(d: AffineDistribution):
    _standard_dev = _standard_dev_impl(d.dist)
    return jtu.tree_map(lambda _sd: _sd * d.scale, _standard_dev)


@_skewness_impl.dispatch
def _skewness(d: AffineDistribution):
    _skewness = _skewness_impl(d.dist)
    return jtu.tree_map(lambda _sk: _sk * jnp.sign(d.scale), _skewness)


@_kurtosis_impl.dispatch
def _kurtosis(d: AffineDistribution):
    _kurtosis = _kurtosis_impl(d.dist)
    return _kurtosis


@_entropy_impl.dispatch
def _entropy1(d: ContinuousAffineDistribution):
    _entropy = _entropy_impl(d.dist)
    return jtu.tree_map(lambda _e: _e + jnp.log(jnp.abs(d.scale)), _entropy)


@_entropy_impl.dispatch
def _entropy2(d: DiscreteAffineDistribution):
    _entropy = _entropy_impl(d.dist)
    return _entropy


@_mgf_impl.dispatch
def _mgf(d: AffineDistribution, t: PyTreeVar):
    exp_mu_t = jnp.exp(t * d.loc)
    sigma_t = t * d.scale
    _mgf = _mgf_impl(d.dist, sigma_t)
    return jtu.tree_map(lambda _m: exp_mu_t * _m, _mgf)


@_cf_impl.dispatch
def _cf(d: AffineDistribution, t: ArrayLike):
    exp_mu_t = jnp.exp(t * d.loc * 1j)
    sigma_t = t * d.scale * 1j
    _cf = _cf_impl(d.dist, sigma_t)
    return jtu.tree_map(lambda _m: exp_mu_t * _m, _cf)


@_logpdf_impl.dispatch
def _logpdf_general(d: AffineDistribution, x: ArrayLike):
    x = (x - d.loc) / d.scale
    _logpdf = _logpdf_impl(d.dist, x)
    return jtu.tree_map(lambda _lp: _lp - jnp.log(jnp.abs(d.scale)), _logpdf)


@_logpdf_impl.dispatch
def _logpdf1(d: ContinuousAffineDistribution, x: ArrayLike):
    x = (x - d.loc) / d.scale
    _logpdf = _logpdf_impl(d.dist, x)
    return jtu.tree_map(lambda _lp: _lp - jnp.log(jnp.abs(d.scale)), _logpdf)


@_logpdf_impl.dispatch
def _logpdf2(d: DiscreteAffineDistribution, x: ArrayLike):
    x = (x - d.loc) / d.scale
    _logpdf = _logpdf_impl(d.dist, x)
    return _logpdf


@_pdf_impl.dispatch
def _pdf_general(d: AffineDistribution, x: ArrayLike):
    abs_scale = jnp.abs(d.scale)
    x = (x - d.loc) / abs_scale
    _pdf = _pdf_impl(d.dist, x)
    return jtu.tree_map(lambda _p: _p / abs_scale, _pdf)


@_pdf_impl.dispatch
def _pdf1(d: ContinuousAffineDistribution, x: ArrayLike):
    abs_scale = jnp.abs(d.scale)
    x = (x - d.loc) / abs_scale
    _pdf = _pdf_impl(d.dist, x)
    return jtu.tree_map(lambda _p: _p / abs_scale, _pdf)


@_pdf_impl.dispatch
def _pdf2(d: DiscreteAffineDistribution, x: ArrayLike):
    x = (x - d.loc) / d.scale
    _pdf = _pdf_impl(d.dist, x)
    return _pdf
