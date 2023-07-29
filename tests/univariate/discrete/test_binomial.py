import numpy as np
import pytest

from fenbux import (
    Binomial,
    cdf,
    kurtois,
    logcdf,
    logpmf,
    mean,
    pmf,
    quantile,
    sf,
    skewness,
    standard_dev,
    variance,
)
from fenbux.scipy_stats import binom


@pytest.mark.parametrize("n, p", [(1, 0.1), (10, 0.5), (50, 0.9)])
def test_mean(n, p):
    dist = Binomial(n, p)
    np.testing.assert_allclose(mean(dist), binom(n, p).mean())


@pytest.mark.parametrize("n, p", [(1, 0.1), (10, 0.5), (50, 0.9)])
def test_variance(n, p):
    dist = Binomial(n, p)
    np.testing.assert_allclose(variance(dist), binom(n, p).var())


@pytest.mark.parametrize("n, p", [(1, 0.1), (10, 0.5), (50, 0.9)])
def test_standard_dev(n, p):
    dist = Binomial(n, p)
    np.testing.assert_allclose(standard_dev(dist), binom(n, p).std())


@pytest.mark.parametrize("n, p", [(1, 0.1), (10, 0.5), (50, 0.9)])
def test_skewness(n, p):
    dist = Binomial(n, p)
    np.testing.assert_allclose(skewness(dist), binom(n, p).stats(moments="s"))


@pytest.mark.parametrize("n, p", [(1, 0.1), (10, 0.5), (50, 0.9)])
def test_kurtosis(n, p):
    dist = Binomial(n, p)
    np.testing.assert_allclose(kurtois(dist), binom(n, p).stats(moments="k"))


@pytest.mark.parametrize("n, p", [(1, 0.1), (10, 0.5), (50, 0.9)])
def test_pmf(n, p):
    x = np.random.binomial(n, p, 10000)
    dist = Binomial(n, p)
    np.testing.assert_allclose(pmf(dist, x), binom(n, p).pmf(x))


@pytest.mark.parametrize("n, p", [(1, 0.1), (10, 0.5), (50, 0.9)])
def test_cdf(n, p):
    x = np.random.binomial(n, p, 10000)
    dist = Binomial(n, p)
    np.testing.assert_allclose(cdf(dist, x), binom(n, p).cdf(x), atol=1e-2)


@pytest.mark.parametrize("n, p", [(1, 0.1), (10, 0.5), (50, 0.9)])
def test_quantile(n, p):
    x = np.random.uniform(0.0, 1.0, 10000)
    dist = Binomial(n, p)
    np.testing.assert_allclose(quantile(dist, x), binom(n, p).ppf(x))


@pytest.mark.parametrize("n, p", [(1, 0.1), (10, 0.5), (50, 0.9)])
def test_logpmf(n, p):
    x = np.random.binomial(n, p, 10000)
    dist = Binomial(n, p)
    np.testing.assert_allclose(logpmf(dist, x), binom(n, p).logpmf(x))


@pytest.mark.parametrize("n, p", [(10, 0.5), (50, 0.9)])
def test_logcdf(n, p):
    x = np.random.binomial(n, p, 10000)
    dist = Binomial(n, p)
    np.testing.assert_allclose(logcdf(dist, x), binom(n, p).logcdf(x), atol=1e-2)


@pytest.mark.parametrize("n, p", [(10, 0.5), (50, 0.9)])
def test_sf(n, p):
    x = np.random.binomial(n, p, 10000)
    dist = Binomial(n, p)
    np.testing.assert_allclose(sf(dist, x), binom(n, p).sf(x), atol=1e-2)
