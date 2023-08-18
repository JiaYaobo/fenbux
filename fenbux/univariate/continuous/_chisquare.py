import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

from ...core import (
    _cdf_impl,
    _cf_impl,
    _intialize_params_tree,
    _kurtosis_impl,
    _logcdf_impl,
    _logpdf_impl,
    _mean_impl,
    _mgf_impl,
    _params_impl,
    _pdf_impl,
    _quantile_impl,
    _sf_impl,
    _skewness_impl,
    _standard_dev_impl,
    _support_impl,
    _variance_impl,
    DTypeLikeFloat,
    KeyArray,
    PyTreeVar,
    rand,
    Shape,
)
from ...dist_math.chi2 import (
    chi2_cdf,
    chi2_cf,
    chi2_logcdf,
    chi2_logpdf,
    chi2_mgf,
    chi2_pdf,
    chi2_ppf,
    chi2_sf,
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


@_params_impl.dispatch
def _params(d: Chisquare):
    return (d.df,)


@_support_impl.dispatch
def _domain(d: Chisquare):
    return jtu.tree_map(lambda _: (0.0, jnp.inf), d.df)


@_mean_impl.dispatch
def _mean(d: Chisquare):
    return jtu.tree_map(lambda df: df, d.df)


@_variance_impl.dispatch
def _variance(d: Chisquare):
    return jtu.tree_map(lambda df: 2 * df, d.df)


@_standard_dev_impl.dispatch
def _standard_dev(d: Chisquare):
    return jtu.tree_map(lambda df: jnp.sqrt(2 * df), d.df)


@_skewness_impl.dispatch
def _skewness(d: Chisquare):
    return jtu.tree_map(lambda df: jnp.sqrt(8 / df), d.df)


@_kurtosis_impl.dispatch
def _kurtosis(d: Chisquare):
    return jtu.tree_map(lambda df: 12 / df, d.df)


@_logpdf_impl.dispatch
def _log_pdf(d: Chisquare, x: PyTreeVar):
    return jtu.tree_map(lambda df: _chisquare_log_pdf(x, df), d.df)


@_pdf_impl.dispatch
def _pdf(d: Chisquare, x: PyTreeVar):
    return jtu.tree_map(lambda df: _chisquare_pdf(x, df), d.df)


@_logcdf_impl.dispatch
def _log_cdf(d: Chisquare, x: PyTreeVar):
    return jtu.tree_map(lambda df: _chisquare_log_cdf(x, df), d.df)


@_cdf_impl.dispatch
def _cdf(d: Chisquare, x: PyTreeVar):
    return jtu.tree_map(lambda df: _chisquare_cdf(x, df), d.df)


@_sf_impl.dispatch
def _sf(d: Chisquare, x: PyTreeVar):
    return jtu.tree_map(lambda df: _chisquare_sf(x, df), d.df)


@_quantile_impl.dispatch
def _quantile(d: Chisquare, p: PyTreeVar):
    return jtu.tree_map(lambda df: _chisquare_quantile(p, df), d.df)


@_mgf_impl.dispatch
def _mgf(d: Chisquare, t: PyTreeVar):
    return jtu.tree_map(lambda df: _chisquare_mgf(t, df), d.df)


@_cf_impl.dispatch
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
    return jtu.tree_map(lambda xx: chi2_logpdf(xx, df), x)


def _chisquare_pdf(x, df):
    return jtu.tree_map(lambda xx: chi2_pdf(xx, df), x)


def _chisquare_log_cdf(x, df):
    return jtu.tree_map(lambda xx: chi2_logcdf(xx, df), x)


def _chisquare_cdf(x, df):
    return jtu.tree_map(lambda xx: chi2_cdf(xx, df), x)


def _chisquare_sf(x, df):
    return jtu.tree_map(lambda xx: chi2_sf(xx, df), x)


def _chisquare_quantile(p, df):
    return jtu.tree_map(lambda pp: chi2_ppf(pp, df), p)


def _chisquare_mgf(t, df):
    return jtu.tree_map(lambda tt: chi2_mgf(tt, df), t)


def _chisquare_cf(t, df):
    return jtu.tree_map(lambda tt: chi2_cf(tt, df), t)
