import numpy as np
import pytest

from fenbux import Poisson
from fenbux.core import (
    cdf,
    kurtosis,
    logcdf,
    logpmf,
    mean,
    pmf,
    skewness,
    standard_dev,
    variance,
)
from fenbux.scipy_stats import poisson


@pytest.mark.parametrize("rate", [0.1, 1.0, 10.0, 20.0])
def test_mean(rate):
    dist = Poisson(rate)
    np.testing.assert_allclose(mean(dist), poisson(rate).mean())


@pytest.mark.parametrize("rate", [0.1, 1.0, 10.0, 20.0])
def test_variance(rate):
    dist = Poisson(rate)
    np.testing.assert_allclose(variance(dist), poisson(rate).var())


@pytest.mark.parametrize("rate", [0.1, 1.0, 10.0, 20.0])
def test_standard_dev(rate):
    dist = Poisson(rate)
    np.testing.assert_allclose(standard_dev(dist), poisson(rate).std())


@pytest.mark.parametrize("rate", [0.1, 1.0, 10.0, 20.0])
def test_skewness(rate):
    dist = Poisson(rate)
    np.testing.assert_allclose(skewness(dist), poisson(rate).stats(moments="s"))


@pytest.mark.parametrize("rate", [0.1, 1.0, 10.0, 20.0])
def test_kurtois(rate):
    dist = Poisson(rate)
    np.testing.assert_allclose(kurtosis(dist), poisson(rate).stats(moments="k"))


@pytest.mark.parametrize("rate", [0.1, 1.0, 10.0, 20.0])
def test_logpmf(rate):
    x = np.random.poisson(rate, 10000)
    n = Poisson(rate)
    np.testing.assert_allclose(logpmf(n, x), poisson(rate).logpmf(x))


@pytest.mark.parametrize("rate", [0.1, 1.0, 10.0, 20.0])
def test_pmf(rate):
    x = np.random.poisson(rate, 10000)
    n = Poisson(rate)
    np.testing.assert_allclose(pmf(n, x), poisson(rate).pmf(x))


@pytest.mark.parametrize("rate", [0.1, 1.0, 10.0, 20.0])
def test_logcdf(rate):
    x = np.random.poisson(rate, 10000)
    n = Poisson(rate)
    np.testing.assert_allclose(logcdf(n, x), poisson(rate).logcdf(x))


@pytest.mark.parametrize("rate", [0.1, 1.0, 10.0, 20.0])
def test_cdf(rate):
    x = np.random.poisson(rate, 10000)
    n = Poisson(rate)
    np.testing.assert_allclose(cdf(n, x), poisson(rate).cdf(x))
