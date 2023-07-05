import numpy as np
import scipy

from fenbux.base import (
    cdf,
    logpdf,
    params,
    pdf,
    quantile,
)
from fenbux.continuous import Normal


def test_params():
    n = Normal(1.0, 2.0)
    np.testing.assert_allclose(params(n), (1.0, 2.0))


def test_logpdf():
    x = np.random.normal(1.0, 2.0, 10000)
    n = Normal(1.0, 2.0)
    np.testing.assert_allclose(logpdf(n, x), scipy.stats.norm(1.0, 2.0).logpdf(x))


def test_pdf():
    x = np.random.normal(1.0, 2.0, 10000)
    n = Normal(1.0, 2.0)
    np.testing.assert_allclose(pdf(n, x), scipy.stats.norm(1.0, 2.0).pdf(x))


def test_cdf():
    x = np.random.normal(1.0, 2.0, 10000)
    n = Normal(1.0, 2.0)
    np.testing.assert_allclose(cdf(n, x), scipy.stats.norm(1.0, 2.0).cdf(x))


def test_quantile():
    x = np.random.uniform(0.0, 1.0, 10000)
    n = Normal(1.0, 2.0)
    np.testing.assert_allclose(quantile(n, x), scipy.stats.norm(1.0, 2.0).ppf(x))
