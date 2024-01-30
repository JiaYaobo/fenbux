import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import ArrayLike

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
    DTypeLikeFloat,
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
from ...tree_utils import tree_map_dist_at
from .._base import ContinuousUnivariateDistribution


class StudentT(ContinuousUnivariateDistribution):
    """Student's t distribution.

    Args:
        df (PyTree): Degrees of freedom.
        dtype (jax.numpy.dtype): dtype of the distribution, default jnp.float_.
        use_batch (bool): Whether to use with vmap. Default False.

    Examples:
        >>> import jax.numpy as jnp
        >>> from fenbux import logpdf
        >>> from fenbux.univariate import StudentT
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
def _logpdf(d: StudentT, x: ArrayLike):
    return tree_map_dist_at(t_logpdf, d, x)


@_pdf_impl.dispatch
def _pdf(d: StudentT, x: ArrayLike):
    return tree_map_dist_at(t_pdf, d, x)


@_logcdf_impl.dispatch
def _logcdf(d: StudentT, x: ArrayLike):
    return tree_map_dist_at(t_logcdf, d, x)


@_cdf_impl.dispatch
def _cdf(d: StudentT, x: ArrayLike):
    return tree_map_dist_at(t_cdf, d, x)


@_quantile_impl.dispatch
def _quantile(d: StudentT, x: ArrayLike):
    return tree_map_dist_at(t_ppf, d, x)


@_sf_impl.dispatch
def _sf(d: StudentT, x: ArrayLike):
    return tree_map_dist_at(t_sf, d, x)


@_rand_impl.dispatch
def _rand(d: StudentT, key: KeyArray, shape: Shape = (), dtype: DTypeLikeFloat = float):
    _key_tree = split_tree(key, d.df)
    return jtu.tree_map(lambda df, k: jr.t(k, df, shape, dtype), d.df, _key_tree)
