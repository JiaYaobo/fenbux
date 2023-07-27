import jax.numpy as jnp
import jax.tree_util as jtu

from ..base import AbstractDistribution, PyTreeVar
from ..base._func import (
    affine,
    cf,
    entropy,
    kurtois,
    logpdf,
    mean,
    mgf,
    params,
    pdf,
    skewness,
    standard_dev,
    variance,
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


@affine.dispatch
def _affine1(d: ContinuousUnivariateDistribution, loc, scale):
    return ContinuousAffineDistribution(loc, scale, d)


@affine.dispatch
def _affine2(d: DiscreteUnivariateDistribution, loc, scale):
    return DiscreteAffineDistribution(loc, scale, d)


@params.dispatch
def _params(d: AffineDistribution):
    return (d.loc, d.scale, params(d.d))


@mean.dispatch
def _mean(d: AffineDistribution):
    _mean = mean(d.d)
    return jtu.tree_map(
        lambda _m, _loc, _scale: _m * _scale + _loc, _mean, d.loc, d.scale
    )


@variance.dispatch
def _variance(d: AffineDistribution):
    _variance = variance(d.d)
    return jtu.tree_map(lambda _v, _scale: _v * _scale**2, _variance, d.scale)


@standard_dev.dispatch
def _standard_dev(d: AffineDistribution):
    _standard_dev = standard_dev(d.d)
    return jtu.tree_map(lambda _sd, _scale: _sd * _scale, _standard_dev, d.scale)


@skewness.dispatch
def _skewness(d: AffineDistribution):
    _skewness = skewness(d.d)
    return jtu.tree_map(lambda _sk, _scale: _sk * jnp.sign(_scale), _skewness, d.scale)


@kurtois.dispatch
def _kurtois(d: AffineDistribution):
    _kurtois = kurtois(d.d)
    return _kurtois


@entropy.dispatch
def _entropy1(d: ContinuousAffineDistribution):
    _entropy = entropy(d.d)
    return jtu.tree_map(
        lambda _e, _scale: _e + jnp.log(jnp.abs(_scale)), _entropy, d.scale
    )


@entropy.dispatch
def _entropy2(d: DiscreteAffineDistribution):
    _entropy = entropy(d.d)
    return _entropy


@mgf.dispatch
def _mgf(d: AffineDistribution, t: PyTreeVar):
    exp_mu_t = jtu.tree_map(lambda _t, _loc: jnp.exp(_t * _loc), t, d.loc)
    sigma_t = jtu.tree_map(lambda _t, _scale: _t * _scale, t, d.scale)
    _mgf = mgf(d.d, sigma_t)
    return jtu.tree_map(lambda _e, _m: _e * _m, exp_mu_t, _mgf)


@cf.dispatch
def _cf(d: AffineDistribution, t: PyTreeVar):
    exp_mu_t = jtu.tree_map(lambda _t, _loc: jnp.exp(_t * _loc * 1j), t, d.loc)
    sigma_t = jtu.tree_map(lambda _t, _scale: _t * _scale * 1j, t, d.scale)
    _cf = cf(d.d, sigma_t)
    return jtu.tree_map(lambda _e, _m: _e * _m, exp_mu_t, _cf)


@logpdf.dispatch
def _logpdf1(d: ContinuousAffineDistribution, x: PyTreeVar):
    x = jtu.tree_map(lambda _x, _loc, _scale: (_x - _loc) / _scale, x, d.loc, d.scale)
    _logpdf = logpdf(d.d, x)
    return jtu.tree_map(
        lambda _lp, _scale: _lp - jnp.log(jnp.abs(_scale)), _logpdf, d.scale
    )


@logpdf.dispatch
def _logpdf2(d: DiscreteAffineDistribution, x: PyTreeVar):
    x = jtu.tree_map(lambda _x, _loc, _scale: (_x - _loc) / _scale, x, d.loc, d.scale)
    _logpdf = logpdf(d.d, x)
    return _logpdf


@pdf.dispatch
def _pdf1(d: ContinuousAffineDistribution, x: PyTreeVar):
    abs_scale = jtu.tree_map(lambda _scale: jnp.abs(_scale), d.scale)
    x = jtu.tree_map(lambda _x, _loc, _scale: (_x - _loc) / _scale, x, d.loc, d.scale)
    _pdf = pdf(d.d, x)
    return jtu.tree_map(lambda _p, _abs_scale: _p / _abs_scale, _pdf, abs_scale)


@pdf.dispatch
def _pdf2(d: DiscreteAffineDistribution, x: PyTreeVar):
    x = jtu.tree_map(lambda _x, _loc, _scale: (_x - _loc) / _scale, x, d.loc, d.scale)
    _pdf = pdf(d.d, x)
    return _pdf
