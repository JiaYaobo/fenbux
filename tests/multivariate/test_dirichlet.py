import numpy as np
import pytest

from fenbux import (
    logpdf,
    mean,
    pdf,
)
from fenbux.multivariate import Dirichlet
from fenbux.scipy_stats import dirichlet


@pytest.mark.parametrize("alpha", [np.ones(2), np.arange(2) + 1])
def test_logpdf(alpha):
    x = np.random.dirichlet(alpha)
    dist = Dirichlet(alpha)
    np.testing.assert_allclose(logpdf(dist, x), dirichlet(alpha).logpdf(x))


@pytest.mark.parametrize("alpha", [np.ones(2), np.arange(2) + 1])
def test_pdf(alpha):
    x = np.random.dirichlet(alpha)
    dist = Dirichlet(alpha)
    np.testing.assert_allclose(pdf(dist, x), dirichlet(alpha).pdf(x))


@pytest.mark.parametrize("alpha", [np.ones(2), np.arange(2) + 1])
def test_mean(alpha):
    dist = Dirichlet(alpha)
    np.testing.assert_allclose(mean(dist), dirichlet(alpha).mean())
