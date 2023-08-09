import numpy as np
import pytest

from fenbux import Weibull
from fenbux.core import (
    cdf,
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
from fenbux.scipy_stats import weibull_min


@pytest.mark.parametrize(
    "shape, scale",
    [
        (0.1, 1.0),
        (0.1, 10.0),
        (1.0, 1.0),
        (1.0, 10.0),
        (10.0, 1.0),
    ],
)
def test_mean(shape, scale):
    dist = Weibull(shape, scale)
    assert np.allclose(mean(dist), weibull_min(c=shape, scale=scale).mean())


@pytest.mark.parametrize(
    "shape, scale",
    [
        (0.1, 1.0),
        (0.1, 10.0),
        (1.0, 1.0),
        (1.0, 10.0),
        (10.0, 1.0),
    ],
)
def test_variance(shape, scale):
    dist = Weibull(shape, scale)
    assert np.allclose(variance(dist), weibull_min(c=shape, scale=scale).var())


@pytest.mark.parametrize(
    "shape, scale",
    [
        (0.1, 1.0),
        (1.0, 1.0),
        (10.0, 1.0),
    ],
)
def test_standard_dev(shape, scale):
    dist = Weibull(shape, scale)
    assert np.allclose(standard_dev(dist), weibull_min(c=shape, scale=scale).std())


@pytest.mark.parametrize(
    "shape, scale",
    [
        (0.1, 1.0),
        (1.0, 1.0),
        (10.0, 1.0),
        (1.0, 10.0),
        (100.0, 1.0),
    ],
)
def test_skewness(shape, scale):
    dist = Weibull(shape, scale)
    assert np.allclose(
        skewness(dist), weibull_min(c=shape, scale=scale).stats(moments="s")
    )


@pytest.mark.parametrize(
    "shape, scale",
    [
        (0.1, 1.0),
    ],
)
def test_pdf(shape, scale):
    dist = Weibull(shape, scale)
    x = np.random.weibull(shape, size=1000) * scale
    assert np.allclose(pdf(dist, x), weibull_min(c=shape, scale=scale).pdf(x))


@pytest.mark.parametrize(
    "shape, scale",
    [
        (0.1, 1.0),
        (0.1, 10.0),
        (1.0, 1.0),
        (1.0, 10.0),
        (10.0, 1.0),
    ],
)
def test_logpdf(shape, scale):
    dist = Weibull(shape, scale)
    x = np.random.weibull(shape, size=100) * scale
    assert np.allclose(logpdf(dist, x), weibull_min(c=shape, scale=scale).logpdf(x))


@pytest.mark.parametrize(
    "shape, scale",
    [
        (0.1, 1.0),
        (0.1, 10.0),
        (1.0, 1.0),
        (1.0, 10.0),
        (10.0, 1.0),
    ],
)
def test_cdf(shape, scale):
    dist = Weibull(shape, scale)
    x = np.linspace(0.1, 10.0, 100)
    assert np.allclose(cdf(dist, x), weibull_min(c=shape, scale=scale).cdf(x))


@pytest.mark.parametrize(
    "shape, scale",
    [
        (0.1, 1.0),
        (0.1, 10.0),
        (1.0, 1.0),
        (1.0, 10.0),
        (10.0, 1.0),
    ],
)
def test_logcdf(shape, scale):
    dist = Weibull(shape, scale)
    x = np.linspace(0.1, 10.0, 100)
    assert np.allclose(logcdf(dist, x), weibull_min(c=shape, scale=scale).logcdf(x))


@pytest.mark.parametrize(
    "shape, scale",
    [
        (0.1, 1.0),
        (0.1, 10.0),
        (1.0, 1.0),
        (1.0, 10.0),
        (10.0, 1.0),
    ],
)
def test_sf(shape, scale):
    dist = Weibull(shape, scale)
    x = np.random.weibull(shape, size=100) * scale
    assert np.allclose(sf(dist, x), weibull_min(c=shape, scale=scale).sf(x))


@pytest.mark.parametrize(
    "shape, scale",
    [
        (0.1, 1.0),
        (0.1, 10.0),
        (1.0, 1.0),
        (1.0, 10.0),
        (10.0, 1.0),
    ],
)
def test_quantile(shape, scale):
    dist = Weibull(shape, scale)
    x = np.random.uniform(size=100)
    assert np.allclose(quantile(dist, x), weibull_min(c=shape, scale=scale).ppf(x))
