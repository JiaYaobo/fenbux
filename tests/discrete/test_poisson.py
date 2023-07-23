import numpy as np
import pytest
import scipy

from fenbux.base import (
    cdf,
    logcdf,
    logpmf,
    pmf,
)
from fenbux.discrete import Poisson


@pytest.mark.parametrize("rate", [0.1, 1.0, 10.0, 20.0])
def test_pmf(rate):
    x = np.random.poisson(rate, 10000)
    n = Poisson(rate)
    np.testing.assert_allclose(pmf(n, x), scipy.stats.poisson(rate).pmf(x))


@pytest.mark.parametrize("rate", [0.1, 1.0, 10.0, 20.0])
def test_cdf(rate):
    x = np.random.poisson(rate, 10000)
    n = Poisson(rate)
    np.testing.assert_allclose(cdf(n, x), scipy.stats.poisson(rate).cdf(x))
