import numpy as np
import pytest

from fenbux.core import (
    kurtosis,
    logpmf,
    mean,
    pmf,
    skewness,
    standard_dev,
    variance,
)
from fenbux.scipy_stats import betabinom
from fenbux.univariate import BetaBinomial


@pytest.mark.parametrize("n, a, b", [(1, 0.1, 0.1), (10, 0.5, 0.5), (50, 0.9, 0.9)])
def test_mean(n, a, b):
    dist = BetaBinomial(n, a, b)
    np.testing.assert_allclose(mean(dist), betabinom(n, a, b).mean())


@pytest.mark.parametrize("n, a, b", [(1, 0.1, 0.1), (10, 0.5, 0.5), (50, 0.9, 0.9)])
def test_variance(n, a, b):
    dist = BetaBinomial(n, a, b)
    np.testing.assert_allclose(variance(dist), betabinom(n, a, b).var())


@pytest.mark.parametrize("n, a, b", [(1, 0.1, 0.1), (10, 0.5, 0.5), (50, 0.9, 0.9)])
def test_standard_dev(n, a, b):
    dist = BetaBinomial(n, a, b)
    np.testing.assert_allclose(standard_dev(dist), betabinom(n, a, b).std())


@pytest.mark.parametrize("n, a, b", [(1, 0.1, 0.1), (10, 0.5, 0.5), (50, 0.9, 0.9)])
def test_skewness(n, a, b):
    dist = BetaBinomial(n, a, b)
    np.testing.assert_allclose(skewness(dist), betabinom(n, a, b).stats(moments="s"))


@pytest.mark.parametrize("n, a, b", [(1, 0.1, 0.1), (10, 0.5, 0.5), (50, 0.9, 0.9)])
def test_kurtosis(n, a, b):
    dist = BetaBinomial(n, a, b)
    np.testing.assert_allclose(kurtosis(dist), betabinom(n, a, b).stats(moments="k"))


@pytest.mark.parametrize("n, a, b", [(1, 0.1, 0.1), (10, 0.5, 0.5), (50, 0.9, 0.9)])
def test_pmf(n, a, b):
    p = np.random.beta(a, b)
    x = np.random.binomial(n, p, 10000)
    dist = BetaBinomial(n, a, b)
    np.testing.assert_allclose(pmf(dist, x), betabinom(n, a, b).pmf(x), atol=1e-6)


@pytest.mark.parametrize("n, a, b", [(1, 0.1, 0.1), (10, 0.5, 0.5), (50, 0.9, 0.9)])
def test_logpmf(n, a, b):
    p = np.random.beta(a, b)
    x = np.random.binomial(n, p, 10000)
    dist = BetaBinomial(n, a, b)
    np.testing.assert_allclose(logpmf(dist, x), betabinom(n, a, b).logpmf(x), atol=1e-6)
