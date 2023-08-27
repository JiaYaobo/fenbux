import numpy as np
import pytest

from fenbux import Exponential
from fenbux.core import (
    cdf,
    entropy,
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
from fenbux.scipy_stats import expon


@pytest.mark.parametrize("rate", [0.5, 1.0, 2.0, 5.0, 10.0])
def test_mean(rate):
    dist = Exponential(rate)
    np.testing.assert_allclose(mean(dist), expon(loc=0, scale=1 / rate).mean())


@pytest.mark.parametrize("rate", [0.5, 1.0, 2.0, 5.0, 10.0])
def test_variance(rate):
    dist = Exponential(rate)
    np.testing.assert_allclose(variance(dist), expon(loc=0, scale=1 / rate).var())


@pytest.mark.parametrize("rate", [0.5, 1.0, 2.0, 5.0, 10.0])
def test_standard_dev(rate):
    dist = Exponential(rate)
    np.testing.assert_allclose(standard_dev(dist), expon(loc=0, scale=1 / rate).std())


@pytest.mark.parametrize("rate", [0.5, 1.0, 2.0, 5.0, 10.0])
def test_skewness(rate):
    dist = Exponential(rate)
    np.testing.assert_allclose(
        skewness(dist), expon(loc=0, scale=1 / rate).stats(moments="s")
    )


@pytest.mark.parametrize("rate", [0.5, 1.0, 2.0, 5.0, 10.0])
def test_kurtois(rate):
    dist = Exponential(rate)
    np.testing.assert_allclose(
        kurtosis(dist), expon(loc=0, scale=1 / rate).stats(moments="k")
    )


@pytest.mark.parametrize("rate", [0.5, 1.0, 2.0, 5.0, 10.0])
def test_entropy(rate):
    n = Exponential(rate)
    np.testing.assert_allclose(entropy(n), expon(loc=0, scale=1 / rate).entropy())


@pytest.mark.parametrize("rate", [0.5, 1.0, 2.0, 5.0, 10.0])
def test_logpdf(rate):
    x = np.random.exponential(1 / rate, 10000)
    dist = Exponential(rate)
    np.testing.assert_allclose(logpdf(dist, x), expon(loc=0, scale=1 / rate).logpdf(x))


@pytest.mark.parametrize("rate", [0.5, 1.0, 2.0, 5.0, 10.0])
def test_pdf(rate):
    x = np.random.exponential(1 / rate, 10000)
    n = Exponential(rate)
    np.testing.assert_allclose(pdf(n, x), expon(loc=0, scale=1 / rate).pdf(x))


@pytest.mark.parametrize("rate", [0.5, 1.0, 2.0, 5.0, 10.0])
def test_logcdf(rate):
    x = np.random.exponential(1 / rate, 10000)
    n = Exponential(rate)
    np.testing.assert_allclose(logcdf(n, x), expon(loc=0, scale=1 / rate).logcdf(x))


@pytest.mark.parametrize("rate", [0.5, 1.0, 2.0, 5.0, 10.0])
def test_cdf(rate):
    x = np.random.exponential(1 / rate, 10000)
    n = Exponential(rate)
    np.testing.assert_allclose(cdf(n, x), expon(loc=0, scale=1 / rate).cdf(x))


# @pytest.mark.parametrize("rate", [0.5, 1.0, 2.0, 5.0, 10.0])
# @pytest.mark.parametrize("q", [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
# def test_quantile(rate, q):
#     n = Exponential(rate)
#     np.testing.assert_allclose(quantile(n, q), expon(loc=0, scale=1 / rate).ppf(q))


@pytest.mark.parametrize("rate", [0.5, 1.0, 2.0, 5.0, 10.0])
def test_sf(rate):
    x = np.random.exponential(1 / rate, 10000)
    dist = Exponential(rate)
    np.testing.assert_allclose(sf(dist, x), expon(loc=0, scale=1 / rate).sf(x))
