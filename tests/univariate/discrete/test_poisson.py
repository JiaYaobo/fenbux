import numpy as np
import pytest

from fenbux import Poisson
from fenbux.base import (
    cdf,
    logcdf,
    logpmf,
    pmf,
)
from fenbux.scipy_stats import poisson


@pytest.mark.parametrize("rate", [0.1, 1.0, 10.0, 20.0])
def test_pmf(rate):
    x = np.random.poisson(rate, 10000)
    n = Poisson(rate)
    np.testing.assert_allclose(pmf(n, x), poisson(rate).pmf(x))


@pytest.mark.parametrize("rate", [0.1, 1.0, 10.0, 20.0])
def test_cdf(rate):
    x = np.random.poisson(rate, 10000)
    n = Poisson(rate)
    np.testing.assert_allclose(cdf(n, x), poisson(rate).cdf(x))
