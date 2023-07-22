import numpy as np
import scipy

from fenbux.base import (
    cdf,
    pmf,
)
from fenbux.discrete import Poisson


def test_pmf():
    x = np.random.poisson(2.0, 10000)
    n = Poisson(2.0)
    np.testing.assert_allclose(pmf(n, x), scipy.stats.poisson(2.0).pmf(x))


def test_cdf():
    x = np.random.poisson(2.0, 10000)
    n = Poisson(2.0)
    np.testing.assert_allclose(cdf(n, x), scipy.stats.poisson(2.0).cdf(x))
