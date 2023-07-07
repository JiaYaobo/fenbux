import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jax.dtypes import canonicalize_dtype

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


class StudentT(AbstractDistribution):

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
def _params(d: StudentT):
    return jtu.tree_leaves(d)


@domain.dispatch
def _domain(d: StudentT):
    return jtu.tree_map(lambda _: (-jnp.inf, jnp.inf), d)

