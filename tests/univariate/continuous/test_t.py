import numpy as np
import pytest

from fenbux import StudentT
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
from fenbux.scipy_stats import t
from tests.helpers import tol


@pytest.mark.parametrize("df", [10.0, 20.0, 50.0])
def test_mean(df):
    dist = StudentT(df)
    np.testing.assert_allclose(mean(dist), t(df).mean())


@pytest.mark.parametrize("df", [10.0, 20.0, 50.0])
def test_variance(df):
    dist = StudentT(df)
    np.testing.assert_allclose(variance(dist), t(df).var(), atol=tol)


@pytest.mark.parametrize("df", [10.0, 20.0, 50.0])
def test_standard_dev(df):
    dist = StudentT(df)
    np.testing.assert_allclose(standard_dev(dist), t(df).std(), atol=tol)


@pytest.mark.parametrize("df", [10.0, 20.0, 50.0])
def test_skewness(df):
    dist = StudentT(df)
    np.testing.assert_allclose(skewness(dist), t(df).stats(moments="s"))


@pytest.mark.parametrize("df", [10.0, 20.0, 50.0])
def test_kurtois(df):
    dist = StudentT(df)
    np.testing.assert_allclose(kurtosis(dist), t(df).stats(moments="k"))


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


@pytest.mark.parametrize("df", [1.0, 5.0])  # need a more precise version
def test_quantile(df):
    x = np.random.uniform(0.0, 1.0, 10000)
    dist = StudentT(df)
    np.testing.assert_allclose(quantile(dist, x), t(df).ppf(x), atol=1e-6)


@pytest.mark.parametrize("df", [1.0, 10.0, 50.0])
def test_sf(df):
    x = np.random.standard_t(df, 10000)
    dist = StudentT(df)
    np.testing.assert_allclose(sf(dist, x), t(df).sf(x), atol=tol)
