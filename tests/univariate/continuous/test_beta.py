import numpy as np
import pytest

from fenbux import Beta
from fenbux.base import (
    cdf,
    logcdf,
    logpdf,
    pdf,
    quantile,
    sf,
)
from fenbux.scipy_stats import beta
from tests.helpers import tol


@pytest.mark.parametrize("a, b", [(1.0, 1.0), (10.0, 10.0), (50.0, 50.0)])
def test_logpdf(a, b):
    x = np.random.beta(a, b, 10000)
    dist = Beta(a, b)
    np.testing.assert_allclose(logpdf(dist, x), beta(a, b).logpdf(x))


@pytest.mark.parametrize("a, b", [(1.0, 1.0), (10.0, 10.0), (50.0, 50.0)])
def test_logcdf(a, b):
    x = np.random.beta(a, b, 10000)
    dist = Beta(a, b)
    np.testing.assert_allclose(logcdf(dist, x), beta(a, b).logcdf(x), atol=tol)


@pytest.mark.parametrize("a, b", [(1.0, 1.0), (10.0, 10.0), (50.0, 50.0)])
def test_pdf(a, b):
    x = np.random.beta(a, b, 10000)
    dist = Beta(a, b)
    np.testing.assert_allclose(pdf(dist, x), beta(a, b).pdf(x))


@pytest.mark.parametrize("a, b", [(1.0, 1.0), (10.0, 10.0), (50.0, 50.0)])
def test_cdf(a, b):
    x = np.random.beta(a, b, 10000)
    dist = Beta(a, b)
    np.testing.assert_allclose(cdf(dist, x), beta(a, b).cdf(x))


@pytest.mark.parametrize("a, b", [(1.0, 1.0), (10.0, 10.0), (50.0, 50.0)])
def test_quantile(a, b):
    x = np.random.uniform(0.0, 1.0, 10000)
    dist = Beta(a, b)
    np.testing.assert_allclose(quantile(dist, x), beta(a, b).ppf(x))


@pytest.mark.parametrize("a, b", [(1.0, 1.0), (10.0, 10.0), (50.0, 50.0)])
def test_sf(a, b):
    x = np.random.beta(a, b, 10000)
    dist = Beta(a, b)
    np.testing.assert_allclose(sf(dist, x), beta(a, b).sf(x), atol=tol)
