import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import ArrayLike

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
    _rand_impl,
    _sf_impl,
    _skewness_impl,
    _standard_dev_impl,
    _support_impl,
    _variance_impl,
    DTypeLikeFloat,
    KeyArray,
    PyTreeVar,
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
from ...tree_utils import tree_map_dist_at
from .._base import ContinuousUnivariateDistribution


class Chisquare(ContinuousUnivariateDistribution):
    """Chisquare distribution.

    Args:
        df (ArrayLike): Degrees of freedom.
        dtype (jax.numpy.dtype): dtype of the distribution, default jnp.float_.
        use_batch (bool): Whether to use with vmap. Default False.

    Examples:
        >>> import jax.numpy as jnp
        >>> from fenbux import logpdf
        >>> from fenbux.univariate import Chisquare
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
def _support(d: Chisquare):
    return jtu.tree_map(lambda df: jnp.zeros_like(df), d.df), jtu.tree_map(
        lambda df: jnp.full_like(df, jnp.inf), d.df
    )


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
def _log_pdf(d: Chisquare, x: ArrayLike):
    return tree_map_dist_at(chi2_logpdf, d, x)


@_pdf_impl.dispatch
def _pdf(d: Chisquare, x: ArrayLike):
    return tree_map_dist_at(chi2_pdf, d, x)


@_logcdf_impl.dispatch
def _log_cdf(d: Chisquare, x: ArrayLike):
    return tree_map_dist_at(chi2_logcdf, d, x)


@_cdf_impl.dispatch
def _cdf(d: Chisquare, x: ArrayLike):
    return tree_map_dist_at(chi2_cdf, d, x)


@_sf_impl.dispatch
def _sf(d: Chisquare, x: ArrayLike):
    return tree_map_dist_at(chi2_sf, d, x)


@_quantile_impl.dispatch
def _quantile(d: Chisquare, p: ArrayLike):
    return tree_map_dist_at(chi2_ppf, d, p)


@_mgf_impl.dispatch
def _mgf(d: Chisquare, t: ArrayLike):
    return tree_map_dist_at(chi2_mgf, d, t)


@_cf_impl.dispatch
def _cf(d: Chisquare, t: ArrayLike):
    return tree_map_dist_at(chi2_cf, d, t)


@_rand_impl.dispatch
def _rand(
    d: Chisquare, key: KeyArray, shape: Shape = (), dtype = float
):
    _key_tree = split_tree(key, jtu.tree_structure(d.df))

    def _fn(key, df):
        return jr.chisquare(key, df, shape=shape, dtype=dtype)

    return jtu.tree_map(lambda k, df: _fn(k, df), _key_tree, d.df)
