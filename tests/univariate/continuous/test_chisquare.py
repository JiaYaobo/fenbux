import jax.random as jr
import numpy as np
import pytest

from fenbux.core import (
    cdf,
    kurtosis,
    logcdf,
    logpdf,
    mean,
    pdf,
    quantile,
    rand,
    sf,
    skewness,
    standard_dev,
    variance,
)
from fenbux.scipy_stats import chi2
from fenbux.univariate import Chisquare
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
    dist = Chisquare(df)
    np.testing.assert_allclose(logpdf(dist, x), chi2(df).logpdf(x))


@pytest.mark.parametrize("df", [1.0, 10.0, 50.0])
def test_logcdf(df: float):
    x = np.random.chisquare(df, 10000)
    dist = Chisquare(df)
    np.testing.assert_allclose(logcdf(dist, x), chi2(df).logcdf(x), atol=tol)


@pytest.mark.parametrize("df", [1.0, 10.0, 50.0])
def test_pdf(df: float):
    x = np.random.chisquare(df, 10000)
    dist = Chisquare(df)
    np.testing.assert_allclose(pdf(dist, x), chi2(df).pdf(x))


@pytest.mark.parametrize("df", [1.0, 10.0, 50.0])
def test_cdf(df: float):
    x = np.random.chisquare(df, 10000)
    dist = Chisquare(df)
    np.testing.assert_allclose(cdf(dist, x), chi2(df).cdf(x))


@pytest.mark.parametrize("df", [1.0, 10.0, 50.0])
def test_quantile(df: float):
    x = np.random.uniform(0.0, 1.0, 10000)
    dist = Chisquare(df)
    np.testing.assert_allclose(quantile(dist, x), chi2(df).ppf(x))


@pytest.mark.parametrize("df", [1.0, 10.0, 50.0])
def test_sf(df: float):
    x = np.random.chisquare(df, 10000)
    dist = Chisquare(df)
    np.testing.assert_allclose(sf(dist, x), chi2(df).sf(x), atol=tol)


@pytest.mark.parametrize(
    "df, sample_shape", [(1.0, (10,)), (10.0, (10,)), (50.0, (10,))]
)
def test_rand(df, sample_shape):
    key = jr.key(0)
    dist = Chisquare(df)
    rvs = rand(dist, key, sample_shape)
    assert rvs.shape == sample_shape
