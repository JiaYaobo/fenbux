import numpy as np
import pytest

from fenbux import Normal
from fenbux.base import (
    cdf,
    logcdf,
    logpdf,
    mean,
    pdf,
    quantile,
    sf,
    variance,
)
from fenbux.scipy_stats import norm


@pytest.mark.parametrize(
    "mu, sd", [(0.0, 1.0), (0.0, 10.0), (5.0, 10.0), (50.0, 100.0)]
)
def test_mean(mu, sd):
    n = Normal(mu, sd)
    np.testing.assert_allclose(mean(n), norm(mu, sd).mean())


@pytest.mark.parametrize(
    "mu, sd", [(0.0, 1.0), (0.0, 10.0), (5.0, 10.0), (50.0, 100.0)]
)
def test_variance(mu, sd):
    n = Normal(mu, sd)
    np.testing.assert_allclose(variance(n), norm(mu, sd).var())


@pytest.mark.parametrize(
    "mu, sd", [(0.0, 1.0), (0.0, 10.0), (5.0, 10.0), (50.0, 100.0)]
)
def test_logpdf(mu, sd):
    x = np.random.normal(mu, sd, 10000)
    n = Normal(mu, sd)
    np.testing.assert_allclose(logpdf(n, x), norm(mu, sd).logpdf(x))


@pytest.mark.parametrize(
    "mu, sd", [(0.0, 1.0), (0.0, 10.0), (5.0, 10.0), (50.0, 100.0)]
)
def test_pdf(mu, sd):
    x = np.random.normal(mu, sd, 10000)
    n = Normal(mu, sd)
    np.testing.assert_allclose(pdf(n, x), norm(mu, sd).pdf(x))


@pytest.mark.parametrize(
    "mu, sd", [(0.0, 1.0), (0.0, 10.0), (5.0, 10.0), (50.0, 100.0)]
)
def test_logcdf(mu, sd):
    x = np.random.normal(mu, sd, 10000)
    n = Normal(mu, sd)
    np.testing.assert_allclose(logcdf(n, x), norm(mu, sd).logcdf(x))


@pytest.mark.parametrize(
    "mu, sd", [(0.0, 1.0), (0.0, 10.0), (5.0, 10.0), (50.0, 100.0)]
)
def test_cdf(mu, sd):
    x = np.random.normal(mu, sd, 10000)
    n = Normal(mu, sd)
    np.testing.assert_allclose(cdf(n, x), norm(mu, sd).cdf(x))


@pytest.mark.parametrize(
    "mu, sd", [(0.0, 1.0), (0.0, 10.0), (5.0, 10.0), (50.0, 100.0)]
)
def test_quantile(mu, sd):
    x = np.random.uniform(0.0, 1.0, 10000)
    n = Normal(mu, sd)
    np.testing.assert_allclose(quantile(n, x), norm(mu, sd).ppf(x))


@pytest.mark.parametrize(
    "mu, sd", [(0.0, 1.0), (0.0, 10.0), (5.0, 10.0), (50.0, 100.0)]
)
def test_sf(mu, sd):
    x = np.random.normal(mu, sd, 10000)
    n = Normal(mu, sd)
    np.testing.assert_allclose(sf(n, x), norm(mu, sd).sf(x))
