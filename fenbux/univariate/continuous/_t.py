import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

from ...core import (
    _cdf_impl,
    _intialize_params_tree,
    _kurtosis_impl,
    _logcdf_impl,
    _logpdf_impl,
    _mean_impl,
    _params_impl,
    _pdf_impl,
    _quantile_impl,
    _rand_impl,
    _sf_impl,
    _skewness_impl,
    _standard_dev_impl,
    _support_impl,
    _variance_impl,
    KeyArray,
    PyTreeVar,
    Shape,
)
from ...dist_math.t import (
    t_cdf,
    t_logcdf,
    t_logpdf,
    t_pdf,
    t_ppf,
    t_sf,
)
from ...random_utils import split_tree
from .._base import ContinuousUnivariateDistribution


class StudentT(ContinuousUnivariateDistribution):
    """Student's t distribution.

    Args:
        df (PyTree): Degrees of freedom.
        dtype (jax.numpy.dtype): dtype of the distribution, default jnp.float_.
        use_batch (bool): Whether to use with vmap. Default False.

    Examples:
        >>> import jax.numpy as jnp
        >>> from fenbux import StudentT, logpdf
        >>> dist = StudentT(1.0)
        >>> logpdf(dist, jnp.ones((10, )))
    """

    df: PyTreeVar

    def __init__(
        self,
        df: PyTreeVar = 1.0,
        dtype=jnp.float_,
        use_batch=False,
    ):
        self.df = _intialize_params_tree(df, use_batch=use_batch, dtype=dtype)


@_params_impl.dispatch
def _params(d: StudentT):
    return (d.df,)


@_support_impl.dispatch
def _domain(d: StudentT):
    return jtu.tree_map(lambda df: jnp.full_like(df, -jnp.inf), d.df), jtu.tree_map(
        lambda df: jnp.full_like(df, jnp.inf), d.df
    )


@_mean_impl.dispatch
def _mean(d: StudentT):
    return jtu.tree_map(lambda df: jnp.where(df > 1, 0.0, jnp.nan), d.df)


@_variance_impl.dispatch
def _variance(d: StudentT):
    return jtu.tree_map(lambda df: jnp.where(df > 2, df / (df - 2), jnp.nan), d.df)


@_standard_dev_impl.dispatch
def _standard_dev(d: StudentT):
    return jtu.tree_map(
        lambda df: jnp.where(df > 2, jnp.sqrt(df / (df - 2)), jnp.nan), d.df
    )


@_skewness_impl.dispatch
def _skewness(d: StudentT):
    return jtu.tree_map(lambda df: jnp.where(df > 3, 0.0, jnp.nan), d.df)


@_kurtosis_impl.dispatch
def _kurtosis(d: StudentT):
    return jtu.tree_map(lambda df: jnp.where(df > 4, 6 / (df - 4), jnp.nan), d.df)


@_logpdf_impl.dispatch
def _logpdf(d: StudentT, x: PyTreeVar):
    return jtu.tree_map(lambda df: _t_logpdf(x, df), d.df)


@_pdf_impl.dispatch
def _pdf(d: StudentT, x: PyTreeVar):
    return jtu.tree_map(lambda df: _t_pdf(x, df), d.df)


@_logcdf_impl.dispatch
def _logcdf(d: StudentT, x: PyTreeVar):
    return jtu.tree_map(lambda df: _t_log_cdf(x, df), d.df)


@_cdf_impl.dispatch
def _cdf(d: StudentT, x: PyTreeVar):
    return jtu.tree_map(lambda df: _t_cdf(x, df), d.df)


@_quantile_impl.dispatch
def _quantile(d: StudentT, x: PyTreeVar):
    return jtu.tree_map(lambda df: _t_quantile(x, df), d.df)


@_sf_impl.dispatch
def _sf(d: StudentT, x: PyTreeVar):
    return jtu.tree_map(lambda df: _t_sf(x, df), d.df)


@_rand_impl.dispatch
def _rand(d: StudentT, key: KeyArray, shape: Shape = (), dtype=jnp.float_):
    _key_tree = split_tree(key, d.df)
    return jtu.tree_map(lambda df, k: jr.t(k, df, shape, dtype), d.df, _key_tree)


def _t_logpdf(x, df):
    return jtu.tree_map(lambda xx: t_logpdf(xx, df), x)


def _t_pdf(x, df):
    return jtu.tree_map(lambda xx: t_pdf(xx, df), x)


def _t_log_cdf(x, df):
    return jtu.tree_map(lambda xx: t_logcdf(xx, df), x)


def _t_cdf(x, df):
    return jtu.tree_map(lambda xx: t_cdf(xx, df), x)


def _t_sf(x, df):
    return jtu.tree_map(lambda xx: t_sf(xx, df), x)


def _t_quantile(x, df):
    return jtu.tree_map(lambda xx: t_ppf(xx, df), x)
