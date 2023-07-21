import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jax.dtypes import canonicalize_dtype
from jax.scipy.special import betainc, betaln, gammaln, xlogy

from ..base import (
    AbstractDistribution,
    cdf,
    DTypeLikeFloat,
    entropy,
    KeyArray,
    kurtois,
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
from ..extension import fdtri
from ..random_utils import split_tree


class F(AbstractDistribution):
    dfn: PyTreeVar
    dfd: PyTreeVar

    def __init__(self, dfn, dfd, dtype=jnp.float_):
        dtype = canonicalize_dtype(dtype)
        self.dfn = jtu.tree_map(lambda x: jnp.asarray(x, dtype=dtype), dfn)
        self.dfd = jtu.tree_map(lambda x: jnp.asarray(x, dtype=dtype), dfd)


@params.dispatch
def _params(d: F):
    return jtu.tree_leaves(d)


@support.dispatch
def _support(d: F):
    _tree = d.broadcast_params()
    return jtu.tree_map(jnp.broadcast_to, (jnp.array(0), jnp.inf), _tree.dfd)


@mean.dispatch
def _mean(d: F):
    _tree = d.broadcast_params()
    return jtu.tree_map(
        lambda dfd: jnp.where(d > 2, dfd / (dfd - 2), jnp.nan), _tree.dfd
    )


@variance.dispatch
def _variance(d: F):
    _tree = d.broadcast_params()
    return jtu.tree_map(
        lambda dfd, dfn: jnp.where(
            d > 4,
            2 * dfd**2 * (dfd + dfn - 2) / (dfn * (dfd - 2) ** 2 * (dfd - 4)),
            jnp.nan,
        ),
        _tree.dfd,
        _tree.dfn,
    )


@standard_dev.dispatch
def _standard_dev(d: F):
    _tree = d.broadcast_params()
    return jtu.tree_map(
        lambda dfd, dfn: jnp.sqrt(
            jnp.where(
                d > 4,
                2 * dfd**2 * (dfd + dfn - 2) / (dfn * (dfd - 2) ** 2 * (dfd - 4)),
                jnp.nan,
            )
        ),
        _tree.dfd,
        _tree.dfn,
    )


@skewness.dispatch
def _skewness(d: F):
    _tree = d.broadcast_params()
    return jtu.tree_map(
        lambda dfd: jnp.where(
            d > 6,
            (
                2
                * (dfd + _tree.dfn - 2)
                * (dfd - 2)
                * jnp.sqrt(dfd - 4)
                / ((dfd - 6) * jnp.sqrt(_tree.dfn * (dfd + _tree.dfn - 2)))
            ),
            jnp.nan,
        ),
        _tree.dfd,
    )


@kurtois.dispatch
def _kurtois(d: F):
    _tree = d.broadcast_params()
    return jtu.tree_map(
        lambda dfd: jnp.where(
            d > 8,
            (
                12
                * dfd
                * (_tree.dfn * (dfd + _tree.dfn - 2) - 3 * dfd * (dfd - 2))
                / (_tree.dfn * (dfd - 2) * (dfd - 4) * (dfd - 6))
            ),
            jnp.nan,
        ),
        _tree.dfd,
    )


@entropy.dispatch
def _entropy(d: F):
    _tree = d.broadcast_params()
    return jtu.tree_map(
        lambda dfd: (
            1
            + jnp.log(dfd / _tree.dfn)
            + (dfd + _tree.dfn) / (dfd - 2)
            + gammaln((_tree.dfn + 1) / 2)
            - gammaln(_tree.dfn / 2)
            - gammaln(dfd / 2)
            - ((_tree.dfn + dfd) / 2) * jnp.log(2)
        ),
        _tree.dfd,
    )


@logpdf.dispatch
def _log_pdf(d: F, x: PyTreeVar):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda dfn, dfd: _f_log_pdf(dfn, dfd, x), _tree.dfn, _tree.dfd)


@pdf.dispatch
def _pdf(d: F, x: PyTreeVar):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda dfn, dfd: _f_pdf(dfn, dfd, x), _tree.dfn, _tree.dfd)


@cdf.dispatch
def _cdf(d: F, x: PyTreeVar):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda dfn, dfd: _f_cdf(dfn, dfd, x), _tree.dfn, _tree.dfd)


@logcdf.dispatch
def _log_cdf(d: F, x: PyTreeVar):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda dfn, dfd: _f_log_cdf(dfn, dfd, x), _tree.dfn, _tree.dfd)


@quantile.dispatch
def _quantile(d: F, x: PyTreeVar):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda dfn, dfd: _f_quantile(dfn, dfd, x), _tree.dfn, _tree.dfd)


@sf.dispatch
def _sf(d: F, x: PyTreeVar):
    _tree = d.broadcast_params()
    return jtu.tree_map(lambda dfn, dfd: _f_sf(dfn, dfd, x), _tree.dfn, _tree.dfd)


@rand.dispatch
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
    def _fn(dfn, dfd, x):
        return (
            dfd / 2 * jnp.log(dfd)
            + dfn / 2 * jnp.log(dfn)
            + xlogy(dfn / 2 - 1, x)
            - (((dfn + dfd) / 2) * jnp.log(dfd + dfn * x) + betaln(dfn / 2, dfd / 2))
        )

    return jtu.tree_map(lambda xx: _fn(dfn, dfd, xx), x)


def _f_pdf(dfn, dfd, x):
    def _fn(dfn, dfd, x):
        return jnp.exp(
            dfd / 2 * jnp.log(dfd)
            + dfn / 2 * jnp.log(dfn)
            + xlogy(dfn / 2 - 1, x)
            - (((dfn + dfd) / 2) * jnp.log(dfd + dfn * x) + betaln(dfn / 2, dfd / 2))
        )

    return jtu.tree_map(lambda xx: _fn(dfn, dfd, xx), x)


def _f_log_cdf(dfn, dfd, x):
    def _fn(dfn, dfd, x):
        return jnp.log(betainc(dfn / 2, dfd / 2, dfn * x / (dfn * x + dfd)))

    return jtu.tree_map(lambda xx: _fn(dfn, dfd, xx), x)


def _f_cdf(dfn, dfd, x):
    def _fn(dfn, dfd, x):
        return betainc(dfn / 2, dfd / 2, dfn * x / (dfn * x + dfd))

    return jtu.tree_map(lambda xx: _fn(dfn, dfd, xx), x)


def _f_quantile(dfn, dfd, x):
    def _fn(dfn, dfd, x):
        return fdtri(dfn, dfd, x)

    return jtu.tree_map(lambda xx: _fn(dfn, dfd, xx), x)


def _f_sf(dfn, dfd, x):
    def _fn(dfn, dfd, x):
        return 1 - betainc(dfn / 2, dfd / 2, dfn * x / (dfn * x + dfd))

    return jtu.tree_map(lambda xx: _fn(dfn, dfd, xx), x)
