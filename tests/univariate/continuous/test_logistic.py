import jax.random as jr
import numpy as np
import pytest

from fenbux import Logistic
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
from fenbux.scipy_stats import logistic
from tests.helpers import tol


@pytest.mark.parametrize(
    "loc, scale", [(0.0, 1.0), (0.0, 10.0), (10.0, 5.0), (50.0, 50.0)]
)
def test_mean(loc, scale):
    dist = Logistic(loc, scale)
    np.testing.assert_allclose(mean(dist), logistic(loc, scale=scale).mean())


@pytest.mark.parametrize(
    "loc, scale", [(0.0, 1.0), (0.0, 10.0), (10.0, 5.0), (50.0, 50.0)]
)
def test_variance(loc, scale):
    dist = Logistic(loc, scale)
    np.testing.assert_allclose(variance(dist), logistic(loc, scale=scale).var())


@pytest.mark.parametrize(
    "loc, scale", [(0.0, 1.0), (0.0, 10.0), (10.0, 5.0), (50.0, 50.0)]
)
def test_standard_dev(loc, scale):
    dist = Logistic(loc, scale)
    np.testing.assert_allclose(
        standard_dev(dist), logistic(loc, scale=scale).std(), rtol=tol
    )


@pytest.mark.parametrize(
    "loc, scale", [(0.0, 1.0), (0.0, 10.0), (10.0, 5.0), (50.0, 50.0)]
)
def test_skewness(loc, scale):
    dist = Logistic(loc, scale)
    np.testing.assert_allclose(skewness(dist), 0.0, atol=tol)


@pytest.mark.parametrize(
    "loc, scale", [(0.0, 1.0), (0.0, 10.0), (10.0, 5.0), (50.0, 50.0)]
)
def test_kurtosis(loc, scale):
    dist = Logistic(loc, scale)
    np.testing.assert_allclose(kurtosis(dist), 1.2, atol=tol)


@pytest.mark.parametrize(
    "loc, scale", [(0.0, 1.0), (0.0, 10.0), (10.0, 5.0), (50.0, 50.0)]
)
def test_entropy(loc, scale):
    dist = Logistic(loc, scale)
    np.testing.assert_allclose(entropy(dist), logistic(loc, scale=scale).entropy())


@pytest.mark.parametrize(
    "loc, scale", [(0.0, 1.0), (0.0, 10.0), (10.0, 5.0), (50.0, 50.0)]
)
def test_pdf(loc, scale):
    dist = Logistic(loc, scale)
    x = np.random.logistic(loc, scale, size=(10000,))
    np.testing.assert_allclose(
        pdf(dist, x),
        logistic(loc, scale=scale).pdf(x),
    )


@pytest.mark.parametrize(
    "loc, scale", [(0.0, 1.0), (0.0, 10.0), (10.0, 5.0), (50.0, 50.0)]
)
def test_logpdf(loc, scale):
    dist = Logistic(loc, scale)
    x = np.random.logistic(loc, scale, size=(10000,))
    np.testing.assert_allclose(
        logpdf(dist, x),
        logistic(loc, scale=scale).logpdf(x),
    )


@pytest.mark.parametrize(
    "loc, scale", [(0.0, 1.0), (0.0, 10.0), (10.0, 5.0), (50.0, 50.0)]
)
def test_cdf(loc, scale):
    dist = Logistic(loc, scale)
    x = np.random.logistic(loc, scale, size=(10000,))
    np.testing.assert_allclose(
        cdf(dist, x),
        logistic(loc, scale=scale).cdf(x),
    )


@pytest.mark.parametrize(
    "loc, scale", [(0.0, 1.0), (0.0, 10.0), (10.0, 5.0), (50.0, 50.0)]
)
def test_logcdf(loc, scale):
    dist = Logistic(loc, scale)
    x = np.random.logistic(loc, scale, size=(10000,))
    np.testing.assert_allclose(
        logcdf(dist, x),
        logistic(loc, scale=scale).logcdf(x),
    )


@pytest.mark.parametrize("loc, scale", [(0.0, 1.0), (0.0, 10.0), (10.0, 5.0)])
def test_sf(loc, scale):
    dist = Logistic(loc, scale)
    x = np.random.logistic(loc, scale, size=(10000,))
    np.testing.assert_allclose(
        sf(dist, x),
        logistic(loc, scale=scale).sf(x),
    )


@pytest.mark.parametrize("loc, scale", [(0.0, 1.0), (0.0, 10.0), (10.0, 5.0)])
def test_quantile(loc, scale):
    dist = Logistic(loc, scale)
    x = np.random.uniform(size=(10000,))
    np.testing.assert_allclose(
        quantile(dist, x),
        logistic(loc, scale=scale).ppf(x),
    )


@pytest.mark.parametrize("loc, scale, sample_shape", [(0.0, 1.0, (1000,))])
def test_rand(loc, scale, sample_shape):
    key = jr.PRNGKey(0)
    dist = Logistic(loc, scale)
    rvs = rand(dist, key, sample_shape)
    assert rvs.shape == sample_shape