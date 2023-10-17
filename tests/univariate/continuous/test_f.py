import jax.random as jr
import numpy as np
import pytest

from fenbux import F
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
from fenbux.scipy_stats import f
from tests.helpers import tol


@pytest.mark.parametrize(("dfn", "dfd"), [(10.0, 10.0), (50.0, 50.0)])
def test_mean(dfn, dfd):
    dist = F(dfn, dfd)
    np.testing.assert_allclose(mean(dist), f(dfn, dfd).mean())


@pytest.mark.parametrize(("dfn", "dfd"), [(10.0, 10.0), (50.0, 50.0)])
def test_variance(dfn, dfd):
    dist = F(dfn, dfd)
    np.testing.assert_allclose(variance(dist), f(dfn, dfd).var(), atol=tol)


@pytest.mark.parametrize(("dfn", "dfd"), [(10.0, 10.0), (50.0, 50.0)])
def test_standard_dev(dfn, dfd):
    dist = F(dfn, dfd)
    np.testing.assert_allclose(standard_dev(dist), f(dfn, dfd).std(), atol=tol)


@pytest.mark.parametrize(("dfn", "dfd"), [(10.0, 10.0), (50.0, 50.0)])
def test_skewness(dfn, dfd):
    dist = F(dfn, dfd)
    np.testing.assert_allclose(skewness(dist), f(dfn, dfd).stats(moments="s"))


@pytest.mark.parametrize(("dfn", "dfd"), [(10.0, 10.0), (50.0, 50.0)])
def test_kurtois(dfn, dfd):
    dist = F(dfn, dfd)
    np.testing.assert_allclose(kurtosis(dist), f(dfn, dfd).stats(moments="k"))


@pytest.mark.parametrize(
    ("dfn", "dfd"), [(1.0, 1.0), (1.0, 10.0), (10.0, 10.0), (50.0, 50.0)]
)
def test_logpdf(dfn, dfd):
    x = np.random.f(dfn, dfd, 10000)
    n = F(dfn, dfd)
    np.testing.assert_allclose(logpdf(n, x), f(dfn, dfd).logpdf(x))


@pytest.mark.parametrize(
    ("dfn", "dfd"), [(1.0, 1.0), (1.0, 10.0), (10.0, 10.0), (50.0, 50.0)]
)
def test_logcdf(dfn, dfd):
    x = np.random.f(dfn, dfd, 10000)
    n = F(dfn, dfd)
    np.testing.assert_allclose(logcdf(n, x), f(dfn, dfd).logcdf(x), atol=tol)


@pytest.mark.parametrize(
    ("dfn", "dfd"), [(1.0, 1.0), (1.0, 10.0), (10.0, 10.0), (50.0, 50.0)]
)
def test_pdf(dfn, dfd):
    x = np.random.f(dfn, dfd, 10000)
    n = F(dfn, dfd)
    np.testing.assert_allclose(pdf(n, x), f(dfn, dfd).pdf(x))


@pytest.mark.parametrize(
    ("dfn", "dfd"), [(1.0, 1.0), (1.0, 10.0), (10.0, 10.0), (50.0, 50.0)]
)
def test_cdf(dfn, dfd):
    x = np.random.f(dfn, dfd, 10000)
    n = F(dfn, dfd)
    np.testing.assert_allclose(cdf(n, x), f(dfn, dfd).cdf(x))


@pytest.mark.parametrize(
    ("dfn", "dfd"), [(1.0, 1.0), (1.0, 10.0), (10.0, 10.0), (50.0, 50.0)]
)
def test_quantile(dfn, dfd):
    x = np.random.uniform(0.0, 1.0, 10000)
    n = F(dfn, dfd)
    np.testing.assert_allclose(quantile(n, x), f(dfn, dfd).ppf(x))


@pytest.mark.parametrize(
    ("dfn", "dfd"), [(1.0, 1.0), (1.0, 10.0), (10.0, 10.0), (50.0, 50.0)]
)
def test_sf(dfn, dfd):
    x = np.random.f(dfn, dfd, 10000)
    n = F(dfn, dfd)
    np.testing.assert_allclose(sf(n, x), f(dfn, dfd).sf(x), atol=tol)


@pytest.mark.parametrize(
    ("dfn", "dfd", "sample_shape"), [(1.0, 1.0, (1000,)), (1.0, 10.0, (1000,))]
)
def test_rand(dfn, dfd, sample_shape):
    key = jr.key(0)
    dist = F(dfn, dfd)
    rvs = rand(dist, key, sample_shape)
    assert rvs.shape == sample_shape