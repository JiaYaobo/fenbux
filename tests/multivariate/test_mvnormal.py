import numpy as np
import pytest

from fenbux import MultivariateNormal
from fenbux.core import (
    entropy,
    logpdf,
    pdf,
)
from fenbux.scipy_stats import multivariate_normal


@pytest.mark.parametrize("loc", [np.zeros(2), np.ones(2), np.arange(2)])
@pytest.mark.parametrize(
    "cov", [np.eye(2), np.diag(np.arange(2) + 1), np.array([[1, 0.5], [0.5, 1]])]
)
def test_logpdf(loc, cov):
    x = np.random.multivariate_normal(loc, cov, 10000)
    dist = MultivariateNormal(loc, cov)
    np.testing.assert_allclose(logpdf(dist, x), multivariate_normal(loc, cov).logpdf(x))


@pytest.mark.parametrize("loc", [np.zeros(2), np.ones(2), np.arange(2)])
@pytest.mark.parametrize(
    "cov", [np.eye(2), np.diag(np.arange(2) + 1), np.array([[1, 0.5], [0.5, 1]])]
)
def test_pdf(loc, cov):
    x = np.random.multivariate_normal(loc, cov, 10000)
    dist = MultivariateNormal(loc, cov)
    np.testing.assert_allclose(pdf(dist, x), multivariate_normal(loc, cov).pdf(x))


@pytest.mark.parametrize("loc", [np.zeros(2), np.ones(2), np.arange(2)])
@pytest.mark.parametrize(
    "cov", [np.eye(2), np.diag(np.arange(2) + 1), np.array([[1, 0.5], [0.5, 1]])]
)
def test_entropy(loc, cov):
    dist = MultivariateNormal(loc, cov)
    np.testing.assert_allclose(entropy(dist), multivariate_normal(loc, cov).entropy())
