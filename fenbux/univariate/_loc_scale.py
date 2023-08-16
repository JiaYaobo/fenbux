import jax.numpy as jnp
import jax.tree_util as jtu

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
from ._base import ContinuousUnivariateDistribution, DiscreteUnivariateDistribution


class AffineDistribution(AbstractDistribution):
    loc: PyTreeVar
    scale: PyTreeVar
    d: AbstractDistribution

    def __init__(self, loc, scale, d):
        self.loc = loc
        self.scale = scale
        self.d = d


class ContinuousAffineDistribution(AffineDistribution):
    def __init__(self, loc, scale, d):
        super().__init__(loc, scale, d)
        assert isinstance(d, ContinuousUnivariateDistribution)


class DiscreteAffineDistribution(AffineDistribution):
    def __init__(self, loc, scale, d):
        super().__init__(loc, scale, d)
        assert isinstance(d, DiscreteUnivariateDistribution)


@_affine_impl.dispatch
def _affine1(d: ContinuousUnivariateDistribution, loc, scale):
    return ContinuousAffineDistribution(loc, scale, d)


@_affine_impl.dispatch
def _affine2(d: DiscreteUnivariateDistribution, loc, scale):
    return DiscreteAffineDistribution(loc, scale, d)


@_params_impl.dispatch
def _params(d: AffineDistribution):
    return (d.loc, d.scale, _params_impl(d.d))


@_mean_impl.dispatch
def _mean(d: AffineDistribution):
    _mean = _mean_impl(d.d)
    return jtu.tree_map(
        lambda _m, _loc, _scale: _m * _scale + _loc, _mean, d.loc, d.scale
    )


@_variance_impl.dispatch
def _variance(d: AffineDistribution):
    _variance = _variance_impl(d.d)
    return jtu.tree_map(lambda _v, _scale: _v * _scale**2, _variance, d.scale)


@_standard_dev_impl.dispatch
def _standard_dev(d: AffineDistribution):
    _standard_dev = _standard_dev_impl(d.d)
    return jtu.tree_map(lambda _sd, _scale: _sd * _scale, _standard_dev, d.scale)


@_skewness_impl.dispatch
def _skewness(d: AffineDistribution):
    _skewness = _skewness_impl(d.d)
    return jtu.tree_map(lambda _sk, _scale: _sk * jnp.sign(_scale), _skewness, d.scale)


@_kurtosis_impl.dispatch
def _kurtosis(d: AffineDistribution):
    _kurtosis = _kurtosis_impl(d.d)
    return _kurtosis


@_entropy_impl.dispatch
def _entropy1(d: ContinuousAffineDistribution):
    _entropy = _entropy_impl(d.d)
    return jtu.tree_map(
        lambda _e, _scale: _e + jnp.log(jnp.abs(_scale)), _entropy, d.scale
    )


@_entropy_impl.dispatch
def _entropy2(d: DiscreteAffineDistribution):
    _entropy = _entropy_impl(d.d)
    return _entropy


@_mgf_impl.dispatch
def _mgf(d: AffineDistribution, t: PyTreeVar):
    exp_mu_t = jtu.tree_map(lambda _t, _loc: jnp.exp(_t * _loc), t, d.loc)
    sigma_t = jtu.tree_map(lambda _t, _scale: _t * _scale, t, d.scale)
    _mgf = _mgf_impl(d.d, sigma_t)
    return jtu.tree_map(lambda _e, _m: _e * _m, exp_mu_t, _mgf)


@_cf_impl.dispatch
def _cf(d: AffineDistribution, t: PyTreeVar):
    exp_mu_t = jtu.tree_map(lambda _t, _loc: jnp.exp(_t * _loc * 1j), t, d.loc)
    sigma_t = jtu.tree_map(lambda _t, _scale: _t * _scale * 1j, t, d.scale)
    _cf = _cf_impl(d.d, sigma_t)
    return jtu.tree_map(lambda _e, _m: _e * _m, exp_mu_t, _cf)


@_logpdf_impl.dispatch
def _logpdf1(d: ContinuousAffineDistribution, x: PyTreeVar):
    x = jtu.tree_map(lambda _x, _loc, _scale: (_x - _loc) / _scale, x, d.loc, d.scale)
    _logpdf = _logpdf_impl(d.d, x)
    return jtu.tree_map(
        lambda _lp, _scale: _lp - jnp.log(jnp.abs(_scale)), _logpdf, d.scale
    )


@_logpdf_impl.dispatch
def _logpdf2(d: DiscreteAffineDistribution, x: PyTreeVar):
    x = jtu.tree_map(lambda _x, _loc, _scale: (_x - _loc) / _scale, x, d.loc, d.scale)
    _logpdf = _logpdf_impl(d.d, x)
    return _logpdf


@_pdf_impl.dispatch
def _pdf1(d: ContinuousAffineDistribution, x: PyTreeVar):
    abs_scale = jtu.tree_map(lambda _scale: jnp.abs(_scale), d.scale)
    x = jtu.tree_map(lambda _x, _loc, _scale: (_x - _loc) / _scale, x, d.loc, d.scale)
    _pdf = _pdf_impl(d.d, x)
    return jtu.tree_map(lambda _p, _abs_scale: _p / _abs_scale, _pdf, abs_scale)


@_pdf_impl.dispatch
def _pdf2(d: DiscreteAffineDistribution, x: PyTreeVar):
    x = jtu.tree_map(lambda _x, _loc, _scale: (_x - _loc) / _scale, x, d.loc, d.scale)
    _pdf = _pdf_impl(d.d, x)
    return _pdf
