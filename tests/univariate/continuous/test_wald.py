import jax.random as jr
import numpy as np
import pytest

from fenbux import Wald
from fenbux.core import (
    cdf,
    entropy,
    kurtosis,
    logcdf,
    logpdf,
    mean,
    pdf,
    quantile,
    rand,
    sf,
    skewness,
    standard_dev,
    variance,
)
from fenbux.scipy_stats import invgauss


@pytest.mark.parametrize("mu", [0.5, 1.0, 2.0])
def test_mean(mu):
    dist = Wald(mu=mu)
    assert np.allclose(mean(dist), invgauss(mu).mean())


@pytest.mark.parametrize("mu", [0.5, 1.0, 2.0])
def test_variance(mu):
    dist = Wald(mu=mu)
    assert np.allclose(variance(dist), invgauss(mu).var())


@pytest.mark.parametrize("mu", [0.5, 1.0, 2.0])
def test_standard_dev(mu):
    dist = Wald(mu=mu)
    assert np.allclose(standard_dev(dist), invgauss(mu).std())


@pytest.mark.parametrize("mu", [0.5, 1.0, 2.0])
def test_skewness(mu):
    dist = Wald(mu=mu)
    assert np.allclose(skewness(dist), invgauss(mu).stats(moments="s"))


@pytest.mark.parametrize("mu", [0.5, 1.0, 2.0])
def test_kurtosis(mu):
    dist = Wald(mu=mu)
    assert np.allclose(kurtosis(dist), invgauss(mu).stats(moments="k"))


@pytest.mark.parametrize("mu", [0.5, 1.0, 2.0])
def test_logpdf(mu):
    dist = Wald(mu=mu)
    x = np.random.wald(mu, scale=1, size=(10, 10))
    assert np.allclose(logpdf(dist, x), invgauss(mu).logpdf(x))


@pytest.mark.parametrize("mu", [0.5, 1.0, 2.0])
def test_pdf(mu):
    dist = Wald(mu=mu)
    x = np.random.wald(mu, scale=1, size=(10, 10))
    assert np.allclose(pdf(dist, x), invgauss(mu).pdf(x))


@pytest.mark.parametrize("mu", [0.5, 1.0, 2.0])
def test_logcdf(mu):
    dist = Wald(mu=mu)
    x = np.random.wald(mu, scale=1, size=(10, 10))
    assert np.allclose(logcdf(dist, x), invgauss(mu).logcdf(x))


@pytest.mark.parametrize("mu", [0.5, 1.0, 2.0])
def test_cdf(mu):
    dist = Wald(mu=mu)
    x = np.random.wald(mu, scale=1, size=(10, 10))
    assert np.allclose(cdf(dist, x), invgauss(mu).cdf(x))


@pytest.mark.parametrize("mu", [0.5, 1.0, 2.0])
def test_sf(mu):
    dist = Wald(mu=mu)
    x = np.random.wald(mu, scale=1, size=(10, 10))
    assert np.allclose(sf(dist, x), invgauss(mu).sf(x))


@pytest.mark.parametrize("mu", [0.5, 1.0, 2.0])
def test_quantile(mu):
    dist = Wald(mu=mu)
    x = np.random.uniform(size=(10, 10))
    assert np.allclose(quantile(dist, x), invgauss(mu).ppf(x))


@pytest.mark.parametrize(
    "mu, sample_shape", [(0.5, (10, 10)), (1.0, (10, 10)), (2.0, (10, 10))]
)
def test_rand(mu, sample_shape):
    dist = Wald(mu=mu)
    key = jr.key(0)
    rvs = rand(dist, key, sample_shape)
    assert rvs.shape == sample_shape
