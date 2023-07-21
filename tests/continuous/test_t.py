import numpy as np

from fenbux.base import (
    cdf,
    logcdf,
    logpdf,
    params,
    pdf,
    quantile,
    sf,
)
from fenbux.continuous import StudentT
from fenbux.scipy_stats import t
from tests.helpers import tol


def test_params():
    dist = StudentT(1.0)
    np.testing.assert_allclose(params(dist), (1.0,))


def test_logpdf():
    x = np.random.standard_t(1.0, 10000)
    dist = StudentT(1.0)
    np.testing.assert_allclose(logpdf(dist, x), t(1.0).logpdf(x))


def test_logcdf():  
    x = np.random.standard_t(1.0, 10000)
    dist = StudentT(1.0)
    np.testing.assert_allclose(logcdf(dist, x), t(1.0).logcdf(x), atol=tol)


def test_pdf():
    x = np.random.standard_t(1.0, 10000)
    dist = StudentT(1.0)
    np.testing.assert_allclose(pdf(dist, x), t(1.0).pdf(x))


def test_cdf():
    x = np.random.standard_t(1.0, 10000)
    dist = StudentT(1.0)
    np.testing.assert_allclose(cdf(dist, x), t(1.0).cdf(x))


def test_quantile():
    x = np.random.uniform(0.0, 1.0, 10000)
    dist = StudentT(1.0)
    np.testing.assert_allclose(quantile(dist, x), t(1.0).ppf(x))


def test_sf():
    x = np.random.standard_t(1.0, 10000)
    dist = StudentT(1.0)
    np.testing.assert_allclose(sf(dist, x), t(1.0).sf(x), atol=tol)

