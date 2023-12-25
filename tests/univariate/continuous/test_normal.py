import jax.random as jr
import numpy as np
import pytest

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
    variance,
)
from fenbux.scipy_stats import norm
from fenbux.univariate import Normal


@pytest.mark.parametrize(
    "mu, sd", [(0.0, 1.0), (0.0, 10.0), (5.0, 10.0), (50.0, 100.0)]
)
def test_mean(mu, sd):
    dist = Normal(mu, sd)
    np.testing.assert_allclose(mean(dist), norm(mu, sd).mean())


@pytest.mark.parametrize(
    "mu, sd", [(0.0, 1.0), (0.0, 10.0), (5.0, 10.0), (50.0, 100.0)]
)
def test_variance(mu, sd):
    dist = Normal(mu, sd)
    np.testing.assert_allclose(variance(dist), norm(mu, sd).var())


@pytest.mark.parametrize(
    "mu, sd", [(0.0, 1.0), (0.0, 10.0), (5.0, 10.0), (50.0, 100.0)]
)
def test_skewness(mu, sd):
    dist = Normal(mu, sd)
    np.testing.assert_allclose(skewness(dist), norm(mu, sd).stats(moments="s"))


@pytest.mark.parametrize(
    "mu, sd", [(0.0, 1.0), (0.0, 10.0), (5.0, 10.0), (50.0, 100.0)]
)
def test_kurtois(mu, sd):
    dist = Normal(mu, sd)
    np.testing.assert_allclose(kurtosis(dist), norm(mu, sd).stats(moments="k"))


@pytest.mark.parametrize("mu, sd", [(0.0, 1.0), (0.0, 10.0), (5.0, 10.0)])
def test_entropy(mu, sd):
    dist = Normal(mu, sd)
    np.testing.assert_allclose(entropy(dist), norm(mu, sd).entropy())


@pytest.mark.parametrize(
    "mu, sd", [(0.0, 1.0), (0.0, 10.0), (5.0, 10.0), (50.0, 100.0)]
)
def test_logpdf(mu, sd):
    x = np.random.normal(mu, sd, 10000)
    dist = Normal(mu, sd)
    np.testing.assert_allclose(logpdf(dist, x), norm(mu, sd).logpdf(x))


@pytest.mark.parametrize(
    "mu, sd", [(0.0, 1.0), (0.0, 10.0), (5.0, 10.0), (50.0, 100.0)]
)
def test_pdf(mu, sd):
    x = np.random.normal(mu, sd, 10000)
    dist = Normal(mu, sd)
    np.testing.assert_allclose(pdf(dist, x), norm(mu, sd).pdf(x))


@pytest.mark.parametrize(
    "mu, sd", [(0.0, 1.0), (0.0, 10.0), (5.0, 10.0), (50.0, 100.0)]
)
def test_logcdf(mu, sd):
    x = np.random.normal(mu, sd, 10000)
    dist = Normal(mu, sd)
    np.testing.assert_allclose(logcdf(dist, x), norm(mu, sd).logcdf(x))


@pytest.mark.parametrize(
    "mu, sd", [(0.0, 1.0), (0.0, 10.0), (5.0, 10.0), (50.0, 100.0)]
)
def test_cdf(mu, sd):
    x = np.random.normal(mu, sd, 10000)
    dist = Normal(mu, sd)
    np.testing.assert_allclose(cdf(dist, x), norm(mu, sd).cdf(x))


@pytest.mark.parametrize(
    "mu, sd", [(0.0, 1.0), (0.0, 10.0), (5.0, 10.0), (50.0, 100.0)]
)
def test_quantile(mu, sd):
    x = np.random.uniform(0.0, 1.0, 10000)
    dist = Normal(mu, sd)
    np.testing.assert_allclose(quantile(dist, x), norm(mu, sd).ppf(x))


@pytest.mark.parametrize(
    "mu, sd", [(0.0, 1.0), (0.0, 10.0), (5.0, 10.0), (50.0, 100.0)]
)
def test_sf(mu, sd):
    x = np.random.normal(mu, sd, 10000)
    dist = Normal(mu, sd)
    np.testing.assert_allclose(sf(dist, x), norm(mu, sd).sf(x))


@pytest.mark.parametrize("shape", [(100,), (2, 5)])
def test_rand(shape):
    dist = Normal(0.0, 1.0)
    key = jr.key(0)
    rvs = rand(dist, key, shape)
    assert rvs.shape == shape
