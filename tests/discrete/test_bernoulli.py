import numpy as np
import scipy

from fenbux.base import (
    cdf,
    params,
    pmf,
    quantile,
)
from fenbux.discrete import Bernoulli


def test_params():
    n = Bernoulli(0.5)
    np.testing.assert_allclose(params(n), (0.5,))


def test_pmf():
    x = np.random.binomial(1, 0.5, 10000)
    n = Bernoulli(0.5)
    np.testing.assert_allclose(pmf(n, x), scipy.stats.bernoulli(0.5).pmf(x))


def test_cdf():
    x = np.random.binomial(1, 0.5, 10000)
    n = Bernoulli(0.5)
    np.testing.assert_allclose(cdf(n, x), scipy.stats.bernoulli(0.5).cdf(x))


def test_quantile():
    x = np.random.uniform(0.0, 1.0, 10000)
    n = Bernoulli(0.5)
    np.testing.assert_allclose(quantile(n, x), scipy.stats.bernoulli(0.5).ppf(x))