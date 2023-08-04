import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jax.scipy.special import betainc, gammaln
from tensorflow_probability.substrates.jax.math import betaincinv

from ...core import (
    _intialize_params_tree,
    cdf,
    KeyArray,
    kurtosis,
    logcdf,
    logpdf,
    mean,
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


@params.dispatch
def _params(d: StudentT):
    return (d.df,)


@support.dispatch
def _domain(d: StudentT):
    return jtu.tree_map(lambda _: (-jnp.inf, jnp.inf), d.df)


@mean.dispatch
def _mean(d: StudentT):
    return jtu.tree_map(lambda df: jnp.where(df > 1, 0.0, jnp.nan), d.df)


@variance.dispatch
def _variance(d: StudentT):
    return jtu.tree_map(lambda df: jnp.where(df > 2, df / (df - 2), jnp.nan), d.df)


@standard_dev.dispatch
def _standard_dev(d: StudentT):
    return jtu.tree_map(
        lambda df: jnp.where(df > 2, jnp.sqrt(df / (df - 2)), jnp.nan), d.df
    )


@skewness.dispatch
def _skewness(d: StudentT):
    return jtu.tree_map(lambda df: jnp.where(df > 3, 0.0, jnp.nan), d.df)


@kurtosis.dispatch
def _kurtosis(d: StudentT):
    return jtu.tree_map(lambda df: jnp.where(df > 4, 6 / (df - 4), jnp.nan), d.df)


@logpdf.dispatch
def _logpdf(d: StudentT, x: PyTreeVar):
    return jtu.tree_map(lambda df: _t_logpdf(x, df), d.df)


@pdf.dispatch
def _pdf(d: StudentT, x: PyTreeVar):
    _logpdf_val = logpdf(d, x)
    return jtu.tree_map(lambda _lp: jnp.exp(_lp), _logpdf_val)


@logcdf.dispatch
def _logcdf(d: StudentT, x: PyTreeVar):
    return jtu.tree_map(lambda df: _t_log_cdf(x, df), d.df)


@cdf.dispatch
def _cdf(d: StudentT, x: PyTreeVar):
    return jtu.tree_map(lambda df: _t_cdf(x, df), d.df)


@quantile.dispatch
def _quantile(d: StudentT, x: PyTreeVar):
    return jtu.tree_map(lambda df: _t_quantile(x, df), d.df)


@sf.dispatch
def _sf(d: StudentT, x: PyTreeVar):
    return jtu.tree_map(lambda df: _t_sf(x, df), d.df)


@rand.dispatch
def _rand(d: StudentT, key: KeyArray, shape: Shape = (), dtype=jnp.float_):
    _key_tree = split_tree(key, d.df)
    return jtu.tree_map(lambda df, k: jr.t(k, df, shape, dtype), d.df, _key_tree)


def _t_logpdf(x, df):
    def _fn(x, df):
        return (
            gammaln((df + 1) / 2)
            - gammaln(df / 2)
            - 0.5 * jnp.log(df * jnp.pi)
            - 0.5 * (df + 1) * jnp.log1p(x**2 / df)
        )

    return jtu.tree_map(lambda xx: _fn(xx, df), x)


def _t_cdf(x, df):
    def _fn(x, df):
        return 0.5 * (
            1.0 + jnp.sign(x) - jnp.sign(x) * betainc(df / 2, 0.5, df / (df + x**2))
        )

    return jtu.tree_map(lambda xx: _fn(xx, df), x)


def _t_sf(x, df):
    def _fn(x, df):
        return 1 - 0.5 * (
            1.0 + jnp.sign(x) - jnp.sign(x) * betainc(df / 2, 0.5, df / (df + x**2))
        )

    return jtu.tree_map(lambda xx: _fn(xx, df), x)


def _t_quantile(x, df):
    def _fn(x, df):
        beta_val = betaincinv(df / 2, 0.5, 1 - jnp.abs(2 * x - 1))
        return jnp.sqrt(df * (1 - beta_val) / beta_val) * jnp.sign(x - 0.5)

    return jtu.tree_map(lambda xx: _fn(xx, df), x)


def _t_log_cdf(x, df):
    def _fn(x, df):
        return jnp.log(
            0.5
            * (
                1.0
                + jnp.sign(x)
                - jnp.sign(x) * betainc(df / 2, 0.5, df / (df + x**2))
            )
        )

    return jtu.tree_map(lambda xx: _fn(xx, df), x)
