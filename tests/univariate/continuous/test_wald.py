import jax.random as jr
import numpy as np
import pytest

from fenbux import Wald
from fenbux.core import (
    cdf,
    entropy,
    kurtosis,
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
from fenbux.scipy_stats import invgauss


@pytest.mark.parametrize(
    "mu, lam",
    [
        (0.1, 3.0),
        (0.2, 10.0),
        (1.0, 1.0),
    ],
)
def test_mean(mu, lam):
    dist = Wald(mu, lam)
    np.testing.assert_allclose(mean(dist), invgauss(mu).mean())


@pytest.mark.parametrize(
    "mu, lam",
    [
        (0.1, 3.0),
        (0.2, 10.0),
        (1.0, 1.0),
    ],
)
def test_variance(mu, lam):
    dist = Wald(mu, lam)
    np.testing.assert_allclose(
        variance(dist), invgauss(mu, scale=1 / np.sqrt(lam)).var()
    )


@pytest.mark.parametrize(
    "mu, lam",
    [
        (0.1, 3.0),
        (0.2, 10.0),
        (1.0, 1.0),
    ],
)
def test_standard_dev(mu, lam):
    dist = Wald(mu, lam)
    np.testing.assert_allclose(
        standard_dev(dist), invgauss(mu, scale=1 / np.sqrt(lam)).std()
    )
