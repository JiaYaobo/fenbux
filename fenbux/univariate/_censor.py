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
    _params_impl,
    _pdf_impl,
    _pmf_impl,
    _support_impl,
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
    """A distribution censored between lower and upper.
    
    Args:
        lower (ArrayLike): Lower bound of the distribution.
        upper (ArrayLike): Upper bound of the distribution.
        d (AbstractDistribution): The uncensored distribution.
    """

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
def _censor_general(d: TransformedDistribution, lower, upper):
    return CensoredDistribution(lower, upper, d)


@_censor_impl.dispatch
def _censor(d: ContinuousUnivariateDistribution, lower, upper):
    return ContinuousCensoredDistribution(lower, upper, d)


@_censor_impl.dispatch
def _censor(d: DiscreteUnivariateDistribution, lower, upper):
    return DiscreteCensoredDistribution(lower, upper, d)


@_params_impl.dispatch
def _params(d: CensoredDistribution):
    return (d.lower, d.upper, _params_impl(d.uncensored))


@_pdf_impl.dispatch
def _pdf_general(d: CensoredDistribution, x: ArrayLike):
    _pdf = _pdf_impl(d.uncensored, x)
    supp = _support_impl(d.uncensored)
    _pdf = jtu.tree_map(
        lambda _p, _l: jnp.where(x < jnp.maximum(_l, d.lower), 0.0, _p),
        _pdf,
        supp[0],
    )
    _pdf = jtu.tree_map(
        lambda _p, _u: jnp.where(x > jnp.minimum(_u, d.upper), 0.0, _p),
        _pdf,
        supp[1],
    )
    return _pdf


@_pdf_impl.dispatch
def _pdf(d: ContinuousCensoredDistribution, x: ArrayLike):
    _pdf = _pdf_impl(d.uncensored, x)
    supp = _support_impl(d.uncensored)
    _pdf = jtu.tree_map(
        lambda _p, _l: jnp.where(x < jnp.maximum(_l, d.lower), 0.0, _p),
        _pdf,
        supp[0],
    )
    _pdf = jtu.tree_map(
        lambda _p, _u: jnp.where(x > jnp.minimum(_u, d.upper), 0.0, _p),
        _pdf,
        supp[1],
    )
    return _pdf


@_logpdf_impl.dispatch
def _logpdf_general(d: CensoredDistribution, x: ArrayLike):
    _lpdf = _logpdf_impl(d.uncensored, x)
    supp = _support_impl(d.uncensored)
    _lpdf = jtu.tree_map(
        lambda _lp, _l: jnp.where(
            x < jnp.where(x < jnp.maximum(_l, d.lower), 0.0, _lp),
            -jnp.inf,
            _lp,
        ),
        _lpdf,
        supp[0],
    )
    _lpdf = jtu.tree_map(
        lambda _lp, _u: jnp.where(x > jnp.minimum(_u, d.upper), 0.0, _lp),
        _lpdf,
        supp[1],
    )
    return _lpdf


@_logpdf_impl.dispatch
def _logpdf(d: ContinuousCensoredDistribution, x: ArrayLike):
    _lpdf = _logpdf_impl(d.uncensored, x)
    supp = _support_impl(d.uncensored)
    _lpdf = jtu.tree_map(
        lambda _lp, _l: jnp.where(
            x < jnp.where(x < jnp.maximum(_l, d.lower), 0.0, _lp),
            -jnp.inf,
            _lp,
        ),
        _lpdf,
        supp[0],
    )
    _lpdf = jtu.tree_map(
        lambda _lp, _u: jnp.where(x > jnp.minimum(_u, d.upper), 0.0, _lp),
        _lpdf,
        supp[1],
    )
    return _lpdf


@_cdf_impl.dispatch
def _cdf_general(d: CensoredDistribution, x: ArrayLike):
    _cdf = _cdf_impl(d.uncensored, x)
    supp = _support_impl(d.uncensored)
    _cdf = jtu.tree_map(
        lambda _c, _l: jnp.where(x < jnp.maximum(_l, d.lower), 0.0, _c),
        _cdf,
        supp[0],
    )
    _cdf = jtu.tree_map(
        lambda _c, _u: jnp.where(x > jnp.minimum(_u, d.upper), 1.0, _c),
        _cdf,
        supp[1],
    )
    return _cdf


@_cdf_impl.dispatch
def _cdf(d: ContinuousCensoredDistribution, x: ArrayLike):
    _cdf = _cdf_impl(d.uncensored, x)
    supp = _support_impl(d.uncensored)
    _cdf = jtu.tree_map(
        lambda _c, _l: jnp.where(x < jnp.maximum(_l, d.lower), 0.0, _c),
        _cdf,
        supp[0],
    )
    _cdf = jtu.tree_map(
        lambda _c, _u: jnp.where(x > jnp.minimum(_u, d.upper), 1.0, _c),
        _cdf,
        supp[1],
    )
    return _cdf


@_logcdf_impl.dispatch
def _logcdf(d: ContinuousCensoredDistribution, x: ArrayLike):
    _lcdf = _logcdf_impl(d.uncensored, x)
    supp = _support_impl(d.uncensored)
    _lcdf = jtu.tree_map(
        lambda _lc, _l: jnp.where(x < jnp.maximum(_l, d.lower), -jnp.inf, _lc),
        _lcdf,
        supp[0],
    )
    _lcdf = jtu.tree_map(
        lambda _lc, _u: jnp.where(x > jnp.minimum(_u, d.upper), 0.0, _lc),
        _lcdf,
        supp[1],
    )
    return _lcdf


@_pmf_impl.dispatch
def _pmf(d: DiscreteCensoredDistribution, x: ArrayLike):
    _pmf = _pmf_impl(d.uncensored, x)
    supp = _support_impl(d.uncensored)
    _pmf = jtu.tree_map(
        lambda _p, _l: jnp.where(x < jnp.maximum(_l, d.lower), 0.0, _p),
        _pmf,
        supp[0],
    )
    _pmf = jtu.tree_map(
        lambda _p, _u: jnp.where(x > jnp.minimum(_u, d.upper), 0.0, _p),
        _pmf,
        supp[1],
    )
    return _pmf


@_logpmf_impl.dispatch
def _logpmf(d: DiscreteCensoredDistribution, x: ArrayLike):
    _lpmf = _logpmf_impl(d.uncensored, x)
    supp = _support_impl(d.uncensored)
    _lpmf = jtu.tree_map(
        lambda _lp, _l: jnp.where(x < jnp.maximum(_l, d.lower), -jnp.inf, _lp),
        _lpmf,
        supp[0],
    )
    _lpmf = jtu.tree_map(
        lambda _lp, _u: jnp.where(x > jnp.minimum(_u, d.upper), 0.0, _lp),
        _lpmf,
        supp[1],
    )
    return _lpmf
