import numpy as np
import pytest
import scipy

from fenbux import Binomial
from fenbux.base import (
    cdf,
    logcdf,
    logpmf,
    pmf,
    quantile,
    sf,
)


@pytest.mark.parametrize("n, p", [(1, 0.1), (10, 0.5), (50, 0.9)])
def test_pmf(n, p):
    x = np.random.binomial(n, p, 10000)
    dist = Binomial(n, p)
    np.testing.assert_allclose(pmf(dist, x), scipy.stats.binom(n, p).pmf(x))


@pytest.mark.parametrize("n, p", [(1, 0.1), (10, 0.5), (50, 0.9)])
def test_cdf(n, p):
    x = np.random.binomial(n, p, 10000)
    dist = Binomial(n, p)
    np.testing.assert_allclose(cdf(dist, x), scipy.stats.binom(n, p).cdf(x), atol=1e-2)


@pytest.mark.parametrize("n, p", [(1, 0.1), (10, 0.5), (50, 0.9)])
def test_quantile(n, p):
    x = np.random.uniform(0.0, 1.0, 10000)
    dist = Binomial(n, p)
    np.testing.assert_allclose(quantile(dist, x), scipy.stats.binom(n, p).ppf(x))


@pytest.mark.parametrize("n, p", [(1, 0.1), (10, 0.5), (50, 0.9)])
def test_logpmf(n, p):
    x = np.random.binomial(n, p, 10000)
    dist = Binomial(n, p)
    np.testing.assert_allclose(logpmf(dist, x), scipy.stats.binom(n, p).logpmf(x))


@pytest.mark.parametrize("n, p", [(10, 0.5), (50, 0.9)])
def test_logcdf(n, p):
    x = np.random.binomial(n, p, 10000)
    dist = Binomial(n, p)
    np.testing.assert_allclose(
        logcdf(dist, x), scipy.stats.binom(n, p).logcdf(x), atol=1e-2
    )


@pytest.mark.parametrize("n, p", [(10, 0.5), (50, 0.9)])
def test_sf(n, p):
    x = np.random.binomial(n, p, 10000)
    dist = Binomial(n, p)
    np.testing.assert_allclose(sf(dist, x), scipy.stats.binom(n, p).sf(x), atol=1e-2)
