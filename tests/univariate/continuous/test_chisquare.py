import numpy as np
import pytest

from fenbux import Chisquare
from fenbux.core import (
    cdf,
    kurtosis,
    logcdf,
    logpdf,
    mean,
    pdf,
    quantile,
    sf,
    skewness,
    standard_dev,
    variance,
)
from fenbux.scipy_stats import chi2
from tests.helpers import tol


@pytest.mark.parametrize("df", [1.0, 10.0, 50.0])
def test_mean(df: float):
    dist = Chisquare(df)
    np.testing.assert_allclose(mean(dist), chi2(df).mean())


@pytest.mark.parametrize("df", [1.0, 10.0, 50.0])
def test_variance(df: float):
    dist = Chisquare(df)
    np.testing.assert_allclose(variance(dist), chi2(df).var(), atol=tol)


@pytest.mark.parametrize("df", [1.0, 10.0, 50.0])
def test_standard_dev(df: float):
    dist = Chisquare(df)
    np.testing.assert_allclose(standard_dev(dist), chi2(df).std(), atol=tol)


@pytest.mark.parametrize("df", [1.0, 10.0, 50.0])
def test_skewness(df: float):
    dist = Chisquare(df)
    np.testing.assert_allclose(skewness(dist), chi2(df).stats(moments="s"))


@pytest.mark.parametrize("df", [1.0, 10.0, 50.0])
def test_kurtois(df: float):
    dist = Chisquare(df)
    np.testing.assert_allclose(kurtosis(dist), chi2(df).stats(moments="k"))


@pytest.mark.parametrize("df", [1.0, 10.0, 50.0])
def test_logpdf(df: float):
    x = np.random.chisquare(df, 10000)
    n = Chisquare(df)
    np.testing.assert_allclose(logpdf(n, x), chi2(df).logpdf(x))


@pytest.mark.parametrize("df", [1.0, 10.0, 50.0])
def test_logcdf(df: float):
    x = np.random.chisquare(df, 10000)
    n = Chisquare(df)
    np.testing.assert_allclose(logcdf(n, x), chi2(df).logcdf(x), atol=tol)


@pytest.mark.parametrize("df", [1.0, 10.0, 50.0])
def test_pdf(df: float):
    x = np.random.chisquare(df, 10000)
    n = Chisquare(df)
    np.testing.assert_allclose(pdf(n, x), chi2(df).pdf(x))


@pytest.mark.parametrize("df", [1.0, 10.0, 50.0])
def test_cdf(df: float):
    x = np.random.chisquare(df, 10000)
    n = Chisquare(df)
    np.testing.assert_allclose(cdf(n, x), chi2(df).cdf(x))


@pytest.mark.parametrize("df", [1.0, 10.0, 50.0])
def test_quantile(df: float):
    x = np.random.uniform(0.0, 1.0, 10000)
    n = Chisquare(df)
    np.testing.assert_allclose(quantile(n, x), chi2(df).ppf(x))


@pytest.mark.parametrize("df", [1.0, 10.0, 50.0])
def test_sf(df: float):
    x = np.random.chisquare(df, 10000)
    n = Chisquare(df)
    np.testing.assert_allclose(sf(n, x), chi2(df).sf(x), atol=tol)
