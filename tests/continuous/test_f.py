import numpy as np

from fenbux.base import (
    cdf,
    logcdf,
    logpdf,
    pdf,
    quantile,
    sf,
)
from fenbux.continuous import F
from fenbux.scipy_stats import f
from tests.helpers import tol


def test_logpdf():
    x = np.random.f(1.0, 2.0, 10000)
    n = F(1.0, 2.0)
    np.testing.assert_allclose(logpdf(n, x), f(1.0, 2.0).logpdf(x))


def test_logcdf():
    x = np.random.f(1.0, 2.0, 10000)
    n = F(1.0, 2.0)
    np.testing.assert_allclose(logcdf(n, x), f(1.0, 2.0).logcdf(x), atol=tol)


def test_pdf():
    x = np.random.f(1.0, 2.0, 10000)
    n = F(1.0, 2.0)
    np.testing.assert_allclose(pdf(n, x), f(1.0, 2.0).pdf(x))


def test_cdf():
    x = np.random.f(1.0, 2.0, 10000)
    n = F(1.0, 2.0)
    np.testing.assert_allclose(cdf(n, x), f(1.0, 2.0).cdf(x))


def test_quantile():
    x = np.random.uniform(0.0, 1.0, 10000)
    n = F(1.0, 2.0)
    np.testing.assert_allclose(quantile(n, x), f(1.0, 2.0).ppf(x))


def test_sf():
    x = np.random.f(1.0, 2.0, 10000)
    n = F(1.0, 2.0)
    np.testing.assert_allclose(sf(n, x), f(1.0, 2.0).sf(x), atol=tol)
