import numpy as np
import pytest

from fenbux import StudentT
from fenbux.base import (
    cdf,
    logcdf,
    logpdf,
    pdf,
    quantile,
    sf,
)
from fenbux.scipy_stats import t
from tests.helpers import tol


@pytest.mark.parametrize("df", [1.0, 10.0, 50.0])
def test_logpdf(df):
    x = np.random.standard_t(df, 10000)
    dist = StudentT(df)
    np.testing.assert_allclose(logpdf(dist, x), t(df).logpdf(x))


@pytest.mark.parametrize("df", [1.0, 10.0, 50.0])
def test_logcdf(df):
    x = np.random.standard_t(df, 10000)
    dist = StudentT(df)
    np.testing.assert_allclose(logcdf(dist, x), t(df).logcdf(x), atol=tol)


@pytest.mark.parametrize("df", [1.0, 10.0, 50.0])
def test_pdf(df):
    x = np.random.standard_t(df, 10000)
    dist = StudentT(df)
    np.testing.assert_allclose(pdf(dist, x), t(df).pdf(x))


@pytest.mark.parametrize("df", [1.0, 10.0, 50.0])
def test_cdf(df):
    x = np.random.standard_t(df, 10000)
    dist = StudentT(df)
    np.testing.assert_allclose(cdf(dist, x), t(df).cdf(x))


@pytest.mark.parametrize("df", [1.0, 5.0]) # need a more precise version...
def test_quantile(df):
    x = np.random.uniform(0.0, 1.0, 10000)
    dist = StudentT(df)
    np.testing.assert_allclose(quantile(dist, x), t(df).ppf(x), atol=tol)


@pytest.mark.parametrize("df", [1.0, 10.0, 50.0])
def test_sf(df):
    x = np.random.standard_t(df, 10000)
    dist = StudentT(df)
    np.testing.assert_allclose(sf(dist, x), t(df).sf(x), atol=tol)
