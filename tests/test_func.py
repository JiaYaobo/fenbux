import numpy as np
from jax import grad, jit, vmap

from fenbux import Normal
from fenbux.core import _pdf_impl
from fenbux.scipy_stats import norm


dist = Normal(0.0, 1.0)
scipy_dist = norm(0.0, 1.0)


def test_jit():
    np.testing.assert_allclose(jit(_pdf_impl)(dist, 0.0),  scipy_dist.pdf(0.0))

def test_vmap():
    np.testing.assert_allclose(vmap(_pdf_impl, in_axes=(None, 0))(dist, np.zeros(10)),  scipy_dist.pdf(np.zeros(10)))

def test_grad():
    assert float(grad(_pdf_impl)(dist, 0.0).mean) == 0.0
    assert float(grad(_pdf_impl)(dist, 0.0).sd) == -0.3989422804014327
