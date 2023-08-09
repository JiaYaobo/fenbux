import numpy as np
import pytest

from fenbux import Pareto
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
from fenbux.scipy_stats import pareto


@pytest.mark.parametrize(
    "shape, scale",
    [
        (2.0, 2.0),
        (5.0, 10.0),
        (10.0, 5.0),
    ],
)
def test_mean(shape, scale):
    dist = Pareto(shape, scale)
    assert np.allclose(mean(dist), pareto(b=shape, scale=scale).mean())


@pytest.mark.parametrize(
    "shape, scale",
    [
        (3.0, 3.0),
        (5.0, 10.0),
        (10.0, 5.0),
    ],
)
def test_variance(shape, scale):
    dist = Pareto(shape, scale)
    assert np.allclose(variance(dist), pareto(b=shape, scale=scale).var())


@pytest.mark.parametrize(
    "shape, scale",
    [
        (2.0, 2.0),
        (5.0, 10.0),
        (10.0, 5.0),
    ],
)
def test_standard_dev(shape, scale):
    dist = Pareto(shape, scale)
    assert np.allclose(standard_dev(dist), pareto(b=shape, scale=scale).std())


@pytest.mark.parametrize(
    "shape, scale",
    [
        (4.0, 4.0),
        (5.0, 10.0),
        (10.0, 5.0),
    ],
)
def test_skewness(shape, scale):
    dist = Pareto(shape, scale)
    assert np.allclose(skewness(dist), pareto(b=shape, scale=scale).stats(moments="s"))


@pytest.mark.parametrize(
    "shape, scale",
    [
        (5.0, 5.0),
        (5.0, 10.0),
        (10.0, 5.0),
    ],
)
def test_kurtosis(shape, scale):
    dist = Pareto(shape, scale)
    assert np.allclose(kurtosis(dist), pareto(b=shape, scale=scale).stats(moments="k"))


@pytest.mark.parametrize(
    "shape, scale",
    [
        (2.0, 2.0),
        (3.0, 3.0),
        (5.0, 10.0),
        (10.0, 5.0),
    ],
)
def test_entropy(shape, scale):
    dist = Pareto(shape, scale)
    assert np.allclose(entropy(dist), pareto(b=shape, scale=scale).entropy())


@pytest.mark.parametrize(
    "shape, scale",
    [
        (2.0, 2.0),
        (5.0, 10.0),
        (10.0, 5.0),
    ],
)
def test_pdf(shape, scale):
    dist = Pareto(shape, scale)
    x = np.random.pareto(shape, size=1000) + scale
    assert np.allclose(pdf(dist, x), pareto(b=shape, scale=scale).pdf(x))


@pytest.mark.parametrize(
    "shape, scale",
    [
        (2.0, 2.0),
        (5.0, 10.0),
        (10.0, 5.0),
    ],
)
def test_logpdf(shape, scale):
    dist = Pareto(shape, scale)
    x = np.random.pareto(shape, size=1000) + scale
    assert np.allclose(logpdf(dist, x), pareto(b=shape, scale=scale).logpdf(x))


@pytest.mark.parametrize(
    "shape, scale",
    [
        (2.0, 2.0),
        (5.0, 10.0),
        (10.0, 5.0),
    ],
)
def test_cdf(shape, scale):
    dist = Pareto(shape, scale)
    x = np.random.pareto(shape, size=1000) + scale
    assert np.allclose(cdf(dist, x), pareto(b=shape, scale=scale).cdf(x))


@pytest.mark.parametrize(
    "shape, scale",
    [
        (2.0, 2.0),
        (5.0, 10.0),
        (10.0, 5.0),
    ],
)
def test_logcdf(shape, scale):
    dist = Pareto(shape, scale)
    x = np.random.pareto(shape, size=1000) + scale
    assert np.allclose(logcdf(dist, x), pareto(b=shape, scale=scale).logcdf(x))


@pytest.mark.parametrize(
    "shape, scale",
    [
        (2.0, 2.0),
        (5.0, 10.0),
        (10.0, 5.0),
    ],
)
def test_sf(shape, scale):
    dist = Pareto(shape, scale)
    x = np.random.pareto(shape, size=1000) + scale
    assert np.allclose(sf(dist, x), pareto(b=shape, scale=scale).sf(x))


@pytest.mark.parametrize(
    "shape, scale",
    [
        (2.0, 2.0),
        (5.0, 10.0),
        (10.0, 5.0),
    ],
)
def test_quantile(shape, scale):
    dist = Pareto(shape, scale)
    x = np.random.uniform(size=1000)
    assert np.allclose(quantile(dist, x), pareto(b=shape, scale=scale).ppf(x))
