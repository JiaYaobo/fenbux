import numpy as np
import pytest
from jax import jit, pmap, vmap

from fenbux import Normal, pdf
from fenbux.scipy_stats import norm


@pytest.mark.parametrize(
    "mu, sd",
    [(0.0, 1.0), (1.0, 5.0), (10.0, 5.0)],
)
def test_jit(mu, sd):
    x = np.random.normal(mu, sd, size=(1000,))
    dist = Normal(mu, sd)
    scipy_dist = norm(mu, sd)
    np.testing.assert_allclose(jit(pdf)(dist, x), scipy_dist.pdf(x))


@pytest.mark.parametrize(
    "mu, sd",
    [(0.0, 1.0), (1.0, 5.0), (10.0, 5.0)],
)
def test_vmap(mu, sd):
    x = np.random.normal(mu, sd, size=(1000,))
    dist = Normal(mu, sd)
    scipy_dist = norm(mu, sd)
    np.testing.assert_allclose(vmap(pdf, in_axes=(None, 0))(dist, x), scipy_dist.pdf(x))


@pytest.mark.parametrize(
    "mu, sd",
    [(0.0, 1.0), (1.0, 5.0), (10.0, 5.0)],
)
def test_pmap(mu, sd):
    x = np.random.normal(mu, sd, size=(8, 1000))
    dist = Normal(mu, sd)
    scipy_dist = norm(mu, sd)
    np.testing.assert_allclose(pmap(pdf, in_axes=(None, 0))(dist, x), scipy_dist.pdf(x))

