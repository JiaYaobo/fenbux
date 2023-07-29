import numpy as np
import pytest
import scipy

from fenbux import Bernoulli
from fenbux.core import (
    cdf,
    kurtois,
    mean,
    pmf,
    quantile,
    skewness,
    standard_dev,
    variance,
)


@pytest.mark.parametrize("p", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_mean(p):
    dist = Bernoulli(p)
    np.testing.assert_allclose(mean(dist), scipy.stats.bernoulli(p).mean())


@pytest.mark.parametrize("p", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_variance(p):
    dist = Bernoulli(p)
    np.testing.assert_allclose(variance(dist), scipy.stats.bernoulli(p).var())


@pytest.mark.parametrize("p", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_standard_dev(p):
    dist = Bernoulli(p)
    np.testing.assert_allclose(standard_dev(dist), scipy.stats.bernoulli(p).std())


@pytest.mark.parametrize("p", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_skewness(p):
    dist = Bernoulli(p)
    np.testing.assert_allclose(
        skewness(dist), scipy.stats.bernoulli(p).stats(moments="s")
    )


@pytest.mark.parametrize("p", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_kurtois(p):
    dist = Bernoulli(p)
    np.testing.assert_allclose(
        kurtois(dist), scipy.stats.bernoulli(p).stats(moments="k")
    )


@pytest.mark.parametrize("p", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_pmf(p):
    x = np.random.binomial(1, p, 10000)
    dist = Bernoulli(p)
    np.testing.assert_allclose(pmf(dist, x), scipy.stats.bernoulli(p).pmf(x))


@pytest.mark.parametrize("p", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_cdf(p):
    x = np.random.binomial(1, p, 10000)
    dist = Bernoulli(p)
    np.testing.assert_allclose(cdf(dist, x), scipy.stats.bernoulli(p).cdf(x))


@pytest.mark.parametrize("p", [0.0, 0.1, 0.5, 0.9])  # p = 1.0 result in weird ...
def test_quantile(p):
    x = np.random.uniform(0.0, 1.0, 10000)
    dist = Bernoulli(p)
    np.testing.assert_allclose(quantile(dist, x), scipy.stats.bernoulli(p).ppf(x))
