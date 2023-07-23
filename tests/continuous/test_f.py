import numpy as np
import pytest

from fenbux.base import (
    cdf,
    logcdf,
    logpdf,
    pdf,
    quantile,
    sf,
)
from fenbux.continuous import F
from fenbux.scipy_stats import f
from tests.helpers import tol


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
