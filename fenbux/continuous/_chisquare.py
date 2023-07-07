import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jax.dtypes import canonicalize_dtype
from jax.scipy.special import gammainc, gammaln, polygamma
from jax.scipy.stats.gamma import logpdf as _jax_gamma_logpdf
from tensorflow_probability.substrates.jax.math import igammainv

from ..base import (
    AbstractDistribution,
    cdf,
    cf,
    DistributionParam,
    domain,
    entropy,
    kurtois,
    logpdf,
    mean,
    mgf,
    params,
    pdf,
    PyTreeVar,
    quantile,
    rand,
    skewness,
    standard_dev,
    variance,
)
from ..random_utils import split_tree


class Chisquare(AbstractDistribution):
    _df: DistributionParam

    def __init__(self, df: PyTreeVar = 0.0, dtype=jnp.float_):
        dtype = canonicalize_dtype(dtype)
        self._df = DistributionParam(
            jtu.tree_map(lambda x: jnp.asarray(x, dtype=dtype), df)
        )

    @property
    def df(self):
        return self._df.val


@params.dispatch
def _params(d: Chisquare):
    return jtu.tree_leaves(d)


@domain.dispatch
def _domain(d: Chisquare):
    return jtu.tree_map(lambda _: (0.0, jnp.inf), d)
