import numpy as np
import pytest
import scipy

from fenbux import Bernoulli
from fenbux.core import (
    cdf,
    logcdf,
    logpmf,
    pmf,
    quantile,
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
