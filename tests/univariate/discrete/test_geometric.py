import numpy as np
import pytest

from fenbux.core import (
    cdf,
    kurtosis,
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
from fenbux.scipy_stats import geom
from fenbux.univariate import Geometric


@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_mean(p):
    dist = Geometric(p)
    np.testing.assert_allclose(mean(dist), geom(p).mean())


@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_variance(p):
    dist = Geometric(p)
    np.testing.assert_allclose(variance(dist), geom(p).var())


@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_standard_dev(p):
    dist = Geometric(p)
    np.testing.assert_allclose(standard_dev(dist), geom(p).std())


@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_skewness(p):
    dist = Geometric(p)
    np.testing.assert_allclose(skewness(dist), geom(p).stats(moments="s"))


@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_kurtosis(p):
    dist = Geometric(p)
    np.testing.assert_allclose(kurtosis(dist), geom(p).stats(moments="k"))


@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_pmf(p):
    x = np.random.geometric(p, 10000)
    dist = Geometric(p)
    np.testing.assert_allclose(pmf(dist, x), geom(p).pmf(x))


@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_cdf(p):
    x = np.random.geometric(p, 10000)
    dist = Geometric(p)
    np.testing.assert_allclose(cdf(dist, x), geom(p).cdf(x), atol=1e-2)


@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_quantile(p):
    x = np.random.uniform(0.0, 1.0, 10000)
    dist = Geometric(p)
    np.testing.assert_allclose(quantile(dist, x), geom(p).ppf(x))


@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_logpmf(p):
    x = np.random.geometric(p, 10000)
    dist = Geometric(p)
    np.testing.assert_allclose(logpmf(dist, x), geom(p).logpmf(x))


@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_logcdf(p):
    x = np.random.geometric(p, 10000)
    dist = Geometric(p)
    np.testing.assert_allclose(logcdf(dist, x), geom(p).logcdf(x))


@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_sf(p):
    x = np.random.geometric(p, 10000)
    dist = Geometric(p)
    np.testing.assert_allclose(sf(dist, x), geom(p).sf(x))
