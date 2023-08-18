import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

from ...core import (
    _cdf_impl,
    _check_params_equal_tree_strcutre,
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
from ...dist_math.f import (
    f_cdf,
    f_logcdf,
    f_logpdf,
    f_pdf,
    f_ppf,
    f_sf,
)
from ...random_utils import split_tree
from .._base import ContinuousUnivariateDistribution


class F(ContinuousUnivariateDistribution):
    """F distribution.

    Args:
        dfn (PyTree): Degrees of freedom in the numerator.
        dfd (PyTree): Degrees of freedom in the denominator.
        dtype (jax.numpy.dtype): dtype of the distribution, default jnp.float_.
        use_batch (bool): Whether to use with vmap. Default False.

    Examples:
        >>> import jax.numpy as jnp
        >>> from fenbux import F, logpdf
        >>> dist = F(1.0, 1.0)
        >>> logpdf(dist, jnp.ones((10, )))
    """

    dfn: PyTreeVar
    dfd: PyTreeVar

    def __init__(self, dfn, dfd, dtype=jnp.float_, use_batch=False):
        _check_params_equal_tree_strcutre(dfn, dfd, use_batch=use_batch)
        self.dfn, self.dfd = _intialize_params_tree(
            dfn, dfd, use_batch=use_batch, dtype=dtype
        )


@_params_impl.dispatch
def _params(d: F):
    return (d.dfn, d.dfd)


@_support_impl.dispatch
def _support(d: F):
    _tree = d.broadcast_params()
    return jtu.tree_map(jnp.broadcast_to, (jnp.array(0), jnp.inf), _tree.dfd)


@_mean_impl.dispatch
def _mean(d: F):
    _tree = d.broadcast_params()
    return jtu.tree_map(
        lambda dfd: jnp.where(dfd > 2, dfd / (dfd - 2), jnp.nan), _tree.dfd
    )


@_variance_impl.dispatch
def _variance(d: F):
    _tree = d.broadcast_params()
    return jtu.tree_map(
        lambda dfd, dfn: jnp.where(
            dfd > 4,
            2 * dfd**2 * (dfd + dfn - 2) / (dfn * (dfd - 2) ** 2 * (dfd - 4)),
            jnp.nan,
        ),
        _tree.dfd,
        _tree.dfn,
    )


@_standard_dev_impl.dispatch
def _standard_dev(d: F):
    _tree = d.broadcast_params()
    return jtu.tree_map(
        lambda dfd, dfn: jnp.sqrt(
            jnp.where(
                dfd > 4,
                2 * dfd**2 * (dfd + dfn - 2) / (dfn * (dfd - 2) ** 2 * (dfd - 4)),
                jnp.nan,
            )
        ),
        _tree.dfd,
        _tree.dfn,
    )


@_skewness_impl.dispatch
def _skewness(d: F):
    _tree = d.broadcast_params()
    return jtu.tree_map(
        lambda dfd, dfn: jnp.where(
            dfd > 6,
            (
                (2 * dfd + dfn - 2)
                * jnp.sqrt((dfd - 4) * 8)
                / ((dfd - 6) * jnp.sqrt(dfn * (dfd + dfn - 2)))
            ),
            jnp.nan,
        ),
        _tree.dfd,
        _tree.dfn,
    )


@_kurtosis_impl.dispatch
def _kurtosis(d: F):
    _tree = d.broadcast_params()
    return jtu.tree_map(
        lambda dfn, dfd: jnp.where(
            dfd > 8,
            (
                12
                * (dfn * (5 * dfd - 22) * (dfd + dfn - 2) + (dfd - 4) * (dfd - 2) ** 2)
                / (dfn * (dfd - 6) * (dfd - 8) * (dfd + dfn - 2))
            ),
            jnp.nan,
        ),
        _tree.dfd,
        _tree.dfn,
    )


@_logpdf_impl.dispatch
def _log_pdf(d: F, x: PyTreeVar):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda dfn, dfd: _f_log_pdf(dfn, dfd, x), _tree.dfn, _tree.dfd)


@_pdf_impl.dispatch
def _pdf(d: F, x: PyTreeVar):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda dfn, dfd: _f_pdf(dfn, dfd, x), _tree.dfn, _tree.dfd)


@_cdf_impl.dispatch
def _cdf(d: F, x: PyTreeVar):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda dfn, dfd: _f_cdf(dfn, dfd, x), _tree.dfn, _tree.dfd)


@_logcdf_impl.dispatch
def _log_cdf(d: F, x: PyTreeVar):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda dfn, dfd: _f_log_cdf(dfn, dfd, x), _tree.dfn, _tree.dfd)


@_quantile_impl.dispatch
def _quantile(d: F, x: PyTreeVar):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda dfn, dfd: _f_quantile(dfn, dfd, x), _tree.dfn, _tree.dfd)


@_sf_impl.dispatch
def _sf(d: F, x: PyTreeVar):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda dfn, dfd: _f_sf(dfn, dfd, x), _tree.dfn, _tree.dfd)


@_rand_impl.dispatch
def _rand(key: KeyArray, d: F, shape: Shape = (), dtype: DTypeLikeFloat = jnp.float_):
    _tree = d.broadcast_params()
    _key_tree = split_tree(key, _tree.dfd)
    return jtu.tree_map(
        lambda k, dfd, dfn: jr.f(k, dfd, dfn, shape=shape, dtype=dtype),
        _key_tree,
        _tree.dfd,
        _tree.dfn,
    )


def _f_log_pdf(dfn, dfd, x):
    return jtu.tree_map(lambda xx: f_logpdf(xx, dfn, dfd), x)


def _f_pdf(dfn, dfd, x):
    return jtu.tree_map(lambda xx: f_pdf(xx, dfn, dfd), x)


def _f_log_cdf(dfn, dfd, x):
    return jtu.tree_map(lambda xx: f_logcdf(xx, dfn, dfd), x)


def _f_cdf(dfn, dfd, x):
    return jtu.tree_map(lambda xx: f_cdf(xx, dfn, dfd), x)


def _f_quantile(dfn, dfd, x):
    return jtu.tree_map(lambda xx: f_ppf(xx, dfn, dfd), x)


def _f_sf(dfn, dfd, x):
    return jtu.tree_map(lambda xx: f_sf(xx, dfn, dfd), x)
