import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jax.scipy.special import gammainc, gammaln, polygamma
from tensorflow_probability.substrates.jax.math import igammainv

from ...core import (
    _intialize_params_tree,
    cdf,
    cf,
    DTypeLikeFloat,
    KeyArray,
    kurtois,
    logcdf,
    logpdf,
    mean,
    mgf,
    params,
    pdf,
    PyTreeVar,
    quantile,
    rand,
    sf,
    Shape,
    skewness,
    standard_dev,
    support,
    variance,
)
from ...random_utils import split_tree
from .._base import ContinuousUnivariateDistribution
from ._gamma import _gamma_log_pdf


class Chisquare(ContinuousUnivariateDistribution):
    """Chisquare distribution.

    Args:
        df (ArrayLike): Degrees of freedom.
        dtype (jax.numpy.dtype): dtype of the distribution, default jnp.float_.
        use_batch (bool): Whether to use with vmap. Default False.

    Examples:
        >>> import jax.numpy as jnp
        >>> from fenbux import Chisquare, logpdf
        >>> dist = Chisquare(1.0)
        >>> logpdf(dist, jnp.ones((10, )))
    """

    df: PyTreeVar

    def __init__(self, df: PyTreeVar = 0.0, dtype=jnp.float_, use_batch=False):
        self.df = _intialize_params_tree(df, use_batch=use_batch, dtype=dtype)


@params.dispatch
def _params(d: Chisquare):
    return (d.df,)


@support.dispatch
def _domain(d: Chisquare):
    return jtu.tree_map(lambda _: (0.0, jnp.inf), d.df)


@mean.dispatch
def _mean(d: Chisquare):
    return jtu.tree_map(lambda df: df, d.df)


@variance.dispatch
def _variance(d: Chisquare):
    return jtu.tree_map(lambda df: 2 * df, d.df)


@standard_dev.dispatch
def _standard_dev(d: Chisquare):
    return jtu.tree_map(lambda df: jnp.sqrt(2 * df), d.df)


@skewness.dispatch
def _skewness(d: Chisquare):
    return jtu.tree_map(lambda df: jnp.sqrt(8 / df), d.df)


@kurtois.dispatch
def _kurtois(d: Chisquare):
    return jtu.tree_map(lambda df: 12 / df, d.df)


@logpdf.dispatch
def _log_pdf(d: Chisquare, x: PyTreeVar):
    return jtu.tree_map(lambda df: _chisquare_log_pdf(x, df), d.df)


@pdf.dispatch
def _pdf(d: Chisquare, x: PyTreeVar):
    return jtu.tree_map(lambda df: _chisquare_pdf(x, df), d.df)


@logcdf.dispatch
def _log_cdf(d: Chisquare, x: PyTreeVar):
    return jtu.tree_map(lambda df: _chisquare_log_cdf(x, df), d.df)


@cdf.dispatch
def _cdf(d: Chisquare, x: PyTreeVar):
    return jtu.tree_map(lambda df: _chisquare_cdf(x, df), d.df)


@sf.dispatch
def _sf(d: Chisquare, x: PyTreeVar):
    return jtu.tree_map(lambda df: _chisquare_sf(x, df), d.df)


@quantile.dispatch
def _quantile(d: Chisquare, p: PyTreeVar):
    return jtu.tree_map(lambda df: _chisquare_quantile(p, df), d.df)


@mgf.dispatch
def _mgf(d: Chisquare, t: PyTreeVar):
    return jtu.tree_map(lambda df: _chisquare_mgf(t, df), d.df)


@cf.dispatch
def _cf(d: Chisquare, t: PyTreeVar):
    return jtu.tree_map(lambda df: _chisquare_cf(t, df), d.df)


@rand.dispatch
def _rand(
    d: Chisquare, key: KeyArray, shape: Shape = (), dtype: DTypeLikeFloat = jnp.float_
):
    _key_tree = split_tree(key, jtu.tree_structure(d.df))

    def _fn(key, df):
        return jr.chisquare(key, df, shape=shape, dtype=dtype)

    return jtu.tree_map(lambda k, df: _fn(k, df), _key_tree, d.df)


def _chisquare_log_pdf(x, df):
    def _fn(x, df):
        return _gamma_log_pdf(x, df / 2, 1 / 2)

    return jtu.tree_map(lambda xx: _fn(xx, df), x)


def _chisquare_pdf(x, df):
    def _fn(x, df):
        return jnp.exp(_gamma_log_pdf(x, df / 2, 1 / 2))

    return jtu.tree_map(lambda xx: _fn(xx, df), x)


def _chisquare_cdf(x, df):
    def _fn(x, df):
        return gammainc(df / 2, x / 2)

    return jtu.tree_map(lambda xx: _fn(xx, df), x)


def _chisquare_sf(x, df):
    def _fn(x, df):
        return 1 - gammainc(df / 2, x / 2)

    return jtu.tree_map(lambda xx: _fn(xx, df), x)


def _chisquare_quantile(p, df):
    def _fn(p, df):
        return 2 * igammainv(df / 2, p)

    return jtu.tree_map(lambda pp: _fn(pp, df), p)


def _chisquare_mgf(t, df):
    def _fn(t, df):
        return (1 - 2 * t) ** (-df / 2)

    return jtu.tree_map(lambda tt: _fn(tt, df), t)


def _chisquare_cf(t, df):
    def _fn(t, df):
        return (1 - 2j * t) ** (-df / 2)

    return jtu.tree_map(lambda tt: _fn(tt, df), t)


def _chisquare_log_cdf(x, df):
    def _fn(x, df):
        return jnp.log(gammainc(df / 2, x / 2))

    return jtu.tree_map(lambda xx: _fn(xx, df), x)
