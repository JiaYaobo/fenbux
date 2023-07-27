import functools as ft
import math
import operator

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import scipy

from fenbux import (
    Bernoulli,
    Binomial,
    Chisquare,
    F,
    Gamma,
    Normal,
    Poisson,
    StudentT,
    Uniform,
)
from fenbux.core import AbstractDistribution, PyTreeVar
from fenbux.tree_utils import full_pytree, zeros_pytree


_tree = ({"a": None, "b": None}, (None, None), None)

def construct_tree_params(tree):
    return jtu.tree_map(lambda _: 1.0, tree)

if jax.config.jax_enable_x64:  # pyright: ignore
    tol = 1e-12
else:
    tol = 1e-6


def shaped_allclose(x, y, **kwargs):
    """As `jnp.allclose`, except:
    - It also supports PyTree arguments.
    - It mandates that shapes match as well (no broadcasting)
    """
    same_structure = jtu.tree_structure(x) == jtu.tree_structure(y)
    allclose = ft.partial(_shaped_allclose, **kwargs)
    return same_structure and jtu.tree_reduce(
        operator.and_, jtu.tree_map(allclose, x, y), True
    )


def _shaped_allclose(x, y, **kwargs):
    if type(x) is not type(y):
        return False
    if isinstance(x, jax.Array):
        x = np.asarray(x)
        y = np.asarray(y)
    if isinstance(x, np.ndarray):
        if np.issubdtype(x.dtype, np.inexact):
            return (
                x.shape == y.shape
                and x.dtype == y.dtype
                and np.allclose(x, y, **kwargs)
            )
        else:
            return x.shape == y.shape and x.dtype == y.dtype and np.all(x == y)
    elif isinstance(x, jax.ShapeDtypeStruct):
        assert x.shape == y.shape and x.dtype == y.dtype
    else:
        return x == y



class FakeDistribution(AbstractDistribution):
    arg1: PyTreeVar
    arg2: PyTreeVar
    arg3: PyTreeVar

    def __init__(self, arg1, arg2, arg3):
        super().__init__()
        self.arg1 = jtu.tree_map(lambda x: jnp.asarray(x, jnp.floating), arg1)
        self.arg2 = jtu.tree_map(lambda x: jnp.asarray(x, jnp.floating), arg2)
        self.arg3 = jtu.tree_map(lambda x: jnp.asarray(x, jnp.floating), arg3)
    

