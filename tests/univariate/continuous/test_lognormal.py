import jax.random as jr
import numpy as np
import pytest

from fenbux import LogNormal
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
from fenbux.scipy_stats import lognorm


@pytest.mark.parametrize("mu, sd", [(0.0, 1.0), (0.0, 10.0), (5.0, 10.0)])
def test_mean(mu, sd):
    dist = LogNormal(mu, sd)
    np.testing.assert_allclose(mean(dist), lognorm(s=sd, scale=np.exp(mu)).mean())


@pytest.mark.parametrize("mu, sd", [(0.0, 1.0), (0.0, 10.0), (5.0, 10.0)])
def test_variance(mu, sd):
    dist = LogNormal(mu, sd)
    np.testing.assert_allclose(variance(dist), lognorm(s=sd, scale=np.exp(mu)).var())


@pytest.mark.parametrize("mu, sd", [(0.0, 1.0), (0.0, 10.0), (5.0, 10.0)])
def test_skewness(mu, sd):
    n = LogNormal(mu, sd)
    np.testing.assert_allclose(
        skewness(n), lognorm(s=sd, scale=np.exp(mu)).stats(moments="s")
    )


@pytest.mark.parametrize("mu, sd", [(0.0, 1.0), (0.0, 10.0), (5.0, 10.0)])
def test_kurtosis(mu, sd):
    n = LogNormal(mu, sd)
    np.testing.assert_allclose(
        kurtosis(n), lognorm(s=sd, scale=np.exp(mu)).stats(moments="k")
    )


@pytest.mark.parametrize("mu, sd", [(0.0, 1.0), (0.0, 10.0), (5.0, 10.0)])
def test_entropy(mu, sd):
    dist = LogNormal(mu, sd)
    np.testing.assert_allclose(entropy(dist), lognorm(s=sd, scale=np.exp(mu)).entropy())


@pytest.mark.parametrize("mu, sd", [(0.0, 1.0), (0.0, 10.0), (5.0, 10.0)])
def test_pdf(mu, sd):
    dist = LogNormal(mu, sd)
    x = np.random.lognormal(mu, sd, size=(1000,))
    np.testing.assert_allclose(pdf(dist, x), lognorm(s=sd, scale=np.exp(mu)).pdf(x))


@pytest.mark.parametrize("mu, sd", [(0.0, 1.0), (0.0, 10.0), (5.0, 10.0)])
def test_logpdf(mu, sd):
    dist = LogNormal(mu, sd)
    x = np.random.lognormal(mu, sd, size=(1000,))
    np.testing.assert_allclose(
        logpdf(dist, x), lognorm(s=sd, scale=np.exp(mu)).logpdf(x)
    )


@pytest.mark.parametrize("mu, sd", [(0.0, 1.0), (0.0, 10.0), (5.0, 10.0)])
def test_cdf(mu, sd):
    dist = LogNormal(mu, sd)
    x = np.random.lognormal(mu, sd, size=(1000,))
    np.testing.assert_allclose(cdf(dist, x), lognorm(s=sd, scale=np.exp(mu)).cdf(x))


@pytest.mark.parametrize("mu, sd", [(0.0, 1.0), (0.0, 10.0), (5.0, 10.0)])
def test_logcdf(mu, sd):
    dist = LogNormal(mu, sd)
    x = np.random.lognormal(mu, sd, size=(1000,))
    np.testing.assert_allclose(
        logcdf(dist, x), lognorm(s=sd, scale=np.exp(mu)).logcdf(x)
    )


@pytest.mark.parametrize("mu, sd", [(0.0, 1.0), (0.0, 10.0), (5.0, 10.0)])
def test_sf(mu, sd):
    dist = LogNormal(mu, sd)
    x = np.random.lognormal(mu, sd, size=(1000,))
    np.testing.assert_allclose(sf(dist, x), lognorm(s=sd, scale=np.exp(mu)).sf(x))


@pytest.mark.parametrize("mu, sd", [(0.0, 1.0), (0.0, 10.0)])
def test_quantile(mu, sd):
    dist = LogNormal(mu, sd)
    p = np.random.uniform(size=(1000,))
    np.testing.assert_allclose(
        quantile(dist, p), lognorm(s=sd, scale=np.exp(mu)).ppf(p)
    )


@pytest.mark.parametrize(
    "mu, sd, sample_shape", [(0.0, 1.0, (1000,)), (0.0, 10.0, (1000,))]
)
def test_rand(mu, sd, sample_shape):
    dist = LogNormal(mu, sd)
    key = jr.key(0)
    rvs = rand(dist, key, sample_shape)
    assert rvs.shape == sample_shape
