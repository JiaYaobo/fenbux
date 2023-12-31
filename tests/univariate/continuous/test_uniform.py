import jax.random as jr
import numpy as np
import pytest

from fenbux.core import (
    cdf,
    entropy,
    kurtosis,
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
from fenbux.scipy_stats import uniform
from fenbux.univariate import Uniform


@pytest.mark.parametrize(
    "lower, upper", [(0.0, 1.0), (-1.0, 1.0), (-1.0, 0.0), (0.0, 10.0), (-10.0, 10.0)]
)
def test_mean(lower, upper):
    dist = Uniform(lower, upper)
    np.testing.assert_allclose(
        mean(dist), uniform(loc=lower, scale=upper - lower).mean()
    )


@pytest.mark.parametrize(
    "lower, upper", [(0.0, 1.0), (-1.0, 1.0), (-1.0, 0.0), (0.0, 10.0), (-10.0, 10.0)]
)
def test_variance(lower, upper):
    dist = Uniform(lower, upper)
    np.testing.assert_allclose(
        variance(dist), uniform(loc=lower, scale=upper - lower).var()
    )


@pytest.mark.parametrize(
    "lower, upper", [(0.0, 1.0), (-1.0, 1.0), (-1.0, 0.0), (0.0, 10.0), (-10.0, 10.0)]
)
def test_standard_dev(lower, upper):
    dist = Uniform(lower, upper)
    np.testing.assert_allclose(
        standard_dev(dist), uniform(loc=lower, scale=upper - lower).std()
    )


@pytest.mark.parametrize(
    "lower, upper", [(0.0, 1.0), (-1.0, 1.0), (-1.0, 0.0), (0.0, 10.0), (-10.0, 10.0)]
)
def test_skewness(lower, upper):
    dist = Uniform(lower, upper)
    np.testing.assert_allclose(
        skewness(dist), uniform(loc=lower, scale=upper - lower).stats(moments="s")
    )


@pytest.mark.parametrize(
    "lower, upper", [(0.0, 1.0), (-1.0, 1.0), (-1.0, 0.0), (0.0, 10.0), (-10.0, 10.0)]
)
def test_kurtois(lower, upper):
    dist = Uniform(lower, upper)
    np.testing.assert_allclose(
        kurtosis(dist), uniform(loc=lower, scale=upper - lower).stats(moments="k")
    )


@pytest.mark.parametrize(
    "lower, upper", [(0.0, 1.0), (-1.0, 1.0), (-1.0, 0.0), (0.0, 10.0), (-10.0, 10.0)]
)
def test_entropy(lower, upper):
    dist = Uniform(lower, upper)
    np.testing.assert_allclose(
        entropy(dist), uniform(loc=lower, scale=upper - lower).entropy()
    )


@pytest.mark.parametrize(
    "lower, upper", [(0.0, 1.0), (-1.0, 1.0), (-1.0, 0.0), (0.0, 10.0), (-10.0, 10.0)]
)
def test_logpdf(lower, upper):
    x = np.random.uniform(lower, upper, 10000)
    n = Uniform(lower, upper)
    np.testing.assert_allclose(
        logpdf(n, x), uniform(loc=lower, scale=upper - lower).logpdf(x)
    )


@pytest.mark.parametrize(
    "lower, upper", [(0.0, 1.0), (-1.0, 1.0), (-1.0, 0.0), (0.0, 10.0), (-10.0, 10.0)]
)
def test_pdf(lower, upper):
    x = np.random.uniform(lower, upper, 10000)
    n = Uniform(lower, upper)
    np.testing.assert_allclose(
        pdf(n, x), uniform(loc=lower, scale=upper - lower).pdf(x)
    )


@pytest.mark.parametrize(
    "lower, upper", [(0.0, 1.0), (-1.0, 1.0), (-1.0, 0.0), (0.0, 10.0), (-10.0, 10.0)]
)
def test_cdf(lower, upper):
    x = np.random.uniform(lower, upper, 10000)
    dist = Uniform(lower, upper)
    np.testing.assert_allclose(
        cdf(dist, x), uniform(loc=lower, scale=upper - lower).cdf(x)
    )


@pytest.mark.parametrize(
    "lower, upper", [(0.0, 1.0), (-1.0, 1.0), (-1.0, 0.0), (0.0, 10.0), (-10.0, 10.0)]
)
def test_quantile(lower, upper):
    x = np.random.uniform(0.0, 1.0, 10000)
    dist = Uniform(lower, upper)
    np.testing.assert_allclose(
        quantile(dist, x), uniform(loc=lower, scale=upper - lower).ppf(x)
    )


@pytest.mark.parametrize(
    "lower, upper", [(0.0, 1.0), (-1.0, 1.0), (-1.0, 0.0), (0.0, 10.0), (-10.0, 10.0)]
)
def test_sf(lower, upper):
    x = np.random.uniform(lower, upper, 10000)
    dist = Uniform(lower, upper)
    np.testing.assert_allclose(
        sf(dist, x), uniform(loc=lower, scale=upper - lower).sf(x)
    )


@pytest.mark.parametrize(
    "lower, upper, sample_shape", [(0.0, 1.0, (10000,)), (-1.0, 1.0, (10000,))]
)
def test_rand(lower, upper, sample_shape):
    key = jr.PRNGKey(0)
    dist = Uniform(lower, upper)
    rvs = rand(dist, key, sample_shape)
    assert rvs.shape == sample_shape
