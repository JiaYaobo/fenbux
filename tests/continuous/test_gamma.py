import numpy as np

from fenbux.base import (
    cdf,
    logcdf,
    logpdf,
    pdf,
    quantile,
    sf,
)
from fenbux.continuous import Gamma
from fenbux.scipy_stats import gamma
from tests.helpers import tol


def test_logpdf():
    x = np.random.gamma(1.0, 2.0, 10000)
    dist = Gamma(1.0, 2.0)
    np.testing.assert_allclose(logpdf(dist, x), gamma(1.0, scale=1 / 2.0).logpdf(x))


def test_logcdf():
    x = np.random.gamma(1.0, 2.0, 10000)
    dist = Gamma(1.0, 2.0)
    np.testing.assert_allclose(
        logcdf(dist, x), gamma(1.0, scale=1 / 2.0).logcdf(x), atol=tol
    )


def test_pdf():
    x = np.random.gamma(1.0, 2.0, 10000)
    dist = Gamma(1.0, 2.0)
    np.testing.assert_allclose(pdf(dist, x), gamma(1.0, scale=1 / 2.0).pdf(x))


def test_cdf():
    x = np.random.gamma(1.0, 2.0, 10000)
    dist = Gamma(1.0, 2.0)
    np.testing.assert_allclose(cdf(dist, x), gamma(1.0, scale=1 / 2.0).cdf(x))


def test_quantile():
    x = np.random.uniform(0.0, 1.0, 10000)
    dist = Gamma(1.0, 2.0)
    np.testing.assert_allclose(quantile(dist, x), gamma(1.0, scale=1 / 2.0).ppf(x))


def test_sf():
    x = np.random.gamma(1.0, 2.0, 10000)
    dist = Gamma(1.0, 2.0)
    np.testing.assert_allclose(sf(dist, x), gamma(1.0, scale=1 / 2.0).sf(x), atol=tol)
