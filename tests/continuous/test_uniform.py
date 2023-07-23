import numpy as np
import pytest
import scipy

from fenbux.base import (
    cdf,
    logpdf,
    pdf,
    quantile,
    sf,
)
from fenbux.continuous import Uniform


@pytest.mark.parametrize(
    "lower, upper", [(0.0, 1.0), (-1.0, 1.0), (-1.0, 0.0), (0.0, 10.0), (-10.0, 10.0)]
)
def test_logpdf(lower, upper):
    x = np.random.uniform(lower, upper, 10000)
    n = Uniform(lower, upper)
    np.testing.assert_allclose(
        logpdf(n, x), scipy.stats.uniform(loc=lower, scale=upper - lower).logpdf(x)
    )

@pytest.mark.parametrize(
    "lower, upper", [(0.0, 1.0), (-1.0, 1.0), (-1.0, 0.0), (0.0, 10.0), (-10.0, 10.0)]
)
def test_pdf(lower, upper):
    x = np.random.uniform(lower, upper, 10000)
    n = Uniform(lower, upper)
    np.testing.assert_allclose(
        pdf(n, x), scipy.stats.uniform(loc=lower, scale=upper - lower).pdf(x)
    )

@pytest.mark.parametrize(
    "lower, upper", [(0.0, 1.0), (-1.0, 1.0), (-1.0, 0.0), (0.0, 10.0), (-10.0, 10.0)]
)
def test_cdf(lower, upper):
    x = np.random.uniform(lower, upper, 10000)
    n = Uniform(lower, upper)
    np.testing.assert_allclose(
        cdf(n, x), scipy.stats.uniform(loc=lower, scale=upper - lower).cdf(x)
    )

@pytest.mark.parametrize(
    "lower, upper", [(0.0, 1.0), (-1.0, 1.0), (-1.0, 0.0), (0.0, 10.0), (-10.0, 10.0)]
)
def test_quantile(lower, upper):
    x = np.random.uniform(0.0, 1.0, 10000)
    n = Uniform(lower, upper)
    np.testing.assert_allclose(
        quantile(n, x), scipy.stats.uniform(loc=lower, scale=upper - lower).ppf(x)
    )

@pytest.mark.parametrize(
    "lower, upper", [(0.0, 1.0), (-1.0, 1.0), (-1.0, 0.0), (0.0, 10.0), (-10.0, 10.0)]
)
def test_sf(lower, upper):
    x = np.random.uniform(lower, upper, 10000)
    n = Uniform(lower, upper)
    np.testing.assert_allclose(
        sf(n, x), scipy.stats.uniform(loc=lower, scale=upper - lower).sf(x)
    )

