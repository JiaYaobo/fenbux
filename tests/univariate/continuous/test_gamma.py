import numpy as np
import pytest

from fenbux import Gamma
from fenbux.base import (
    cdf,
    kurtois,
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
from fenbux.scipy_stats import gamma
from tests.helpers import tol


@pytest.mark.parametrize(
    "alpha, beta", [(1.0, 1.0), (1.0, 10.0), (10.0, 5.0), (50.0, 50.0)]
)
def test_mean(alpha, beta):
    dist = Gamma(alpha, beta)
    np.testing.assert_allclose(mean(dist), gamma(alpha, scale=1 / beta).mean())


@pytest.mark.parametrize(
    "alpha, beta", [(1.0, 1.0), (1.0, 10.0), (10.0, 5.0), (50.0, 50.0)]
)
def test_variance(alpha, beta):
    dist = Gamma(alpha, beta)
    np.testing.assert_allclose(
        variance(dist), gamma(alpha, scale=1 / beta).var(), atol=tol
    )


@pytest.mark.parametrize(
    "alpha, beta", [(1.0, 1.0), (1.0, 10.0), (10.0, 5.0), (50.0, 50.0)]
)
def test_standard_dev(alpha, beta):
    dist = Gamma(alpha, beta)
    np.testing.assert_allclose(
        standard_dev(dist), gamma(alpha, scale=1 / beta).std(), atol=tol
    )


@pytest.mark.parametrize(
    "alpha, beta", [(1.0, 1.0), (1.0, 10.0), (10.0, 5.0), (50.0, 50.0)]
)
def test_skewness(alpha, beta):
    dist = Gamma(alpha, beta)
    np.testing.assert_allclose(
        skewness(dist), gamma(alpha, scale=1 / beta).stats(moments="s")
    )


@pytest.mark.parametrize(
    "alpha, beta", [(1.0, 1.0), (1.0, 10.0), (10.0, 5.0), (50.0, 50.0)]
)
def test_kurtois(alpha, beta):
    dist = Gamma(alpha, beta)
    np.testing.assert_allclose(
        kurtois(dist), gamma(alpha, scale=1 / beta).stats(moments="k")
    )


@pytest.mark.parametrize(
    "alpha, beta", [(1.0, 1.0), (1.0, 10.0), (10.0, 5.0), (50.0, 50.0)]
)
def test_logpdf(alpha, beta):
    x = np.random.gamma(alpha, beta, 10000)
    dist = Gamma(alpha, beta)
    np.testing.assert_allclose(logpdf(dist, x), gamma(alpha, scale=1 / beta).logpdf(x))


@pytest.mark.parametrize(
    "alpha, beta", [(1.0, 1.0), (1.0, 10.0), (10.0, 5.0), (50.0, 50.0)]
)
def test_logcdf(alpha, beta):
    x = np.random.gamma(alpha, beta, 10000)
    dist = Gamma(alpha, beta)
    np.testing.assert_allclose(
        logcdf(dist, x), gamma(alpha, scale=1 / beta).logcdf(x), atol=tol
    )


@pytest.mark.parametrize(
    "alpha, beta", [(1.0, 1.0), (1.0, 10.0), (10.0, 5.0), (50.0, 50.0)]
)
def test_pdf(alpha, beta):
    x = np.random.gamma(alpha, beta, 10000)
    dist = Gamma(alpha, beta)
    np.testing.assert_allclose(
        pdf(dist, x), gamma(alpha, scale=1 / beta).pdf(x), atol=tol
    )


@pytest.mark.parametrize(
    "alpha, beta", [(1.0, 1.0), (1.0, 10.0), (10.0, 5.0), (50.0, 50.0)]
)
def test_cdf(alpha, beta):
    x = np.random.gamma(alpha, beta, 10000)
    dist = Gamma(alpha, beta)
    np.testing.assert_allclose(cdf(dist, x), gamma(alpha, scale=1 / beta).cdf(x))


@pytest.mark.parametrize(
    "alpha, beta", [(1.0, 1.0), (1.0, 10.0), (10.0, 5.0), (50.0, 50.0)]
)
def test_quantile(alpha, beta):
    x = np.random.uniform(0.0, 1.0, 10000)
    dist = Gamma(alpha, beta)
    np.testing.assert_allclose(quantile(dist, x), gamma(alpha, scale=1 / beta).ppf(x))


@pytest.mark.parametrize(
    "alpha, beta", [(1.0, 1.0), (1.0, 10.0), (10.0, 5.0), (50.0, 50.0)]
)
def test_sf(alpha, beta):
    x = np.random.gamma(alpha, beta, 10000)
    dist = Gamma(alpha, beta)
    np.testing.assert_allclose(
        sf(dist, x), gamma(alpha, scale=1 / beta).sf(x), atol=tol
    )
