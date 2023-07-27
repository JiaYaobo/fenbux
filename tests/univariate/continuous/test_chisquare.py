import numpy as np
import pytest

from fenbux import Chisquare
from fenbux.base import (
    cdf,
    logcdf,
    logpdf,
    pdf,
    quantile,
    sf,
)
from fenbux.scipy_stats import chi2
from tests.helpers import tol


@pytest.mark.parametrize("df", [1.0, 10.0, 50.0])
def test_logpdf(df):
    x = np.random.chisquare(df, 10000)
    n = Chisquare(df)
    np.testing.assert_allclose(logpdf(n, x), chi2(df).logpdf(x))


@pytest.mark.parametrize("df", [1.0, 10.0, 50.0])
def test_logcdf(df):
    x = np.random.chisquare(df, 10000)
    n = Chisquare(df)
    np.testing.assert_allclose(logcdf(n, x), chi2(df).logcdf(x), atol=tol)


@pytest.mark.parametrize("df", [1.0, 10.0, 50.0])
def test_pdf(df):
    x = np.random.chisquare(df, 10000)
    n = Chisquare(df)
    np.testing.assert_allclose(pdf(n, x), chi2(df).pdf(x))


@pytest.mark.parametrize("df", [1.0, 10.0, 50.0])
def test_cdf(df):
    x = np.random.chisquare(df, 10000)
    n = Chisquare(df)
    np.testing.assert_allclose(cdf(n, x), chi2(df).cdf(x))


@pytest.mark.parametrize("df", [1.0, 10.0, 50.0])
def test_quantile(df):
    x = np.random.uniform(0.0, 1.0, 10000)
    n = Chisquare(df)
    np.testing.assert_allclose(quantile(n, x), chi2(df).ppf(x))


@pytest.mark.parametrize("df", [1.0, 10.0, 50.0])
def test_sf(df):
    x = np.random.chisquare(df, 10000)
    n = Chisquare(df)
    np.testing.assert_allclose(sf(n, x), chi2(df).sf(x), atol=tol)
