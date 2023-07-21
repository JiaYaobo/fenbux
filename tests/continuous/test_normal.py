import numpy as np

from fenbux.base import (
    cdf,
    logcdf,
    logpdf,
    mean,
    params,
    pdf,
    quantile,
    sf,
    variance,
)
from fenbux.continuous import Normal
from fenbux.scipy_stats import norm


def test_params():
    n = Normal(1.0, 2.0)
    np.testing.assert_allclose(params(n), (1.0, 2.0))


def test_mean():
    n = Normal(1.0, 2.0)
    np.testing.assert_allclose(mean(n), norm(1.0, 2.0).mean())


def test_variance():
    n = Normal(1.0, 2.0)
    np.testing.assert_allclose(variance(n), norm(1.0, 2.0).var())


def test_logpdf():
    x = np.random.normal(1.0, 2.0, 10000)
    n = Normal(1.0, 2.0)
    np.testing.assert_allclose(logpdf(n, x), norm(1.0, 2.0).logpdf(x))


def test_pdf():
    x = np.random.normal(1.0, 2.0, 10000)
    n = Normal(1.0, 2.0)
    np.testing.assert_allclose(pdf(n, x), norm(1.0, 2.0).pdf(x))


def test_logcdf():
    x = np.random.normal(1.0, 2.0, 10000)
    n = Normal(1.0, 2.0)
    np.testing.assert_allclose(logcdf(n, x), norm(1.0, 2.0).logcdf(x))


def test_cdf():
    x = np.random.normal(1.0, 2.0, 10000)
    n = Normal(1.0, 2.0)
    np.testing.assert_allclose(cdf(n, x), norm(1.0, 2.0).cdf(x))


def test_quantile():
    x = np.random.uniform(0.0, 1.0, 10000)
    n = Normal(1.0, 2.0)
    np.testing.assert_allclose(quantile(n, x), norm(1.0, 2.0).ppf(x))


def test_sf():
    x = np.random.normal(1.0, 2.0, 10000)
    n = Normal(1.0, 2.0)
    np.testing.assert_allclose(sf(n, x), norm(1.0, 2.0).sf(x))
