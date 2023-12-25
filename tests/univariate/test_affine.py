import numpy as np
import pytest

from fenbux import affine, logpdf, mean, pdf, standard_dev, variance
from fenbux.scipy_stats import norm
from fenbux.univariate import Normal


@pytest.mark.parametrize(
    "mu, sd, loc, scale",
    [
        (0.0, 1.0, 0.0, 1.0),
        (0.0, 10.0, 1.0, 1.0),
        (5.0, 10.0, 5.0, 1.0),
        (50.0, 100.0, 10.0, 1.0),
    ],
)
def test_affine_mean(mu, sd, loc, scale):
    dist = Normal(mu, sd)
    a = affine(dist, loc, scale)
    np.testing.assert_allclose(mean(a), norm(mu, sd).mean() + loc)


@pytest.mark.parametrize(
    "mu, sd, loc, scale",
    [
        (0.0, 1.0, 0.0, 1.0),
        (0.0, 10.0, 1.0, 2.0),
        (5.0, 10.0, 5.0, 1.0),
        (50.0, 100.0, 10.0, 2.0),
    ],
)
def test_affine_variance(mu, sd, loc, scale):
    dist = Normal(mu, sd)
    a = affine(dist, loc, scale)
    np.testing.assert_allclose(variance(a), norm(mu, sd).var() * scale**2)


@pytest.mark.parametrize(
    "mu, sd, loc, scale",
    [
        (0.0, 1.0, 0.0, 1.0),
        (0.0, 10.0, 1.0, 2.0),
        (5.0, 10.0, 5.0, 1.0),
        (50.0, 100.0, 10.0, 2.0),
    ],
)
def test_affine_std(mu, sd, loc, scale):
    dist = Normal(mu, sd)
    a = affine(dist, loc, scale)
    np.testing.assert_allclose(standard_dev(a), norm(mu, sd).std() * scale)


@pytest.mark.parametrize(
    "mu, sd, loc, scale",
    [
        (0.0, 1.0, 0.0, 1.0),
        (0.0, 10.0, 1.0, 2.0),
        (5.0, 10.0, 5.0, 1.0),
        (50.0, 100.0, 10.0, 2.0),
    ],
)
def test_affine_logpdf(mu, sd, loc, scale):
    dist = Normal(mu, sd)
    a = affine(dist, loc, scale)
    x = np.linspace(-100, 100, 100)
    np.testing.assert_allclose(
        logpdf(a, x), norm(mu, sd).logpdf((x - loc) / scale) - np.log(scale)
    )


@pytest.mark.parametrize(
    "mu, sd, loc, scale",
    [
        (0.0, 1.0, 0.0, 2.0),
        (0.0, 10.0, 1.0, 1.0),
        (5.0, 10.0, 5.0, 2.0),
        (50.0, 100.0, 10.0, 1.0),
    ],
)
def test_affine_pdf(mu, sd, loc, scale):
    dist = Normal(mu, sd)
    a = affine(dist, loc, scale)
    x = np.random.normal(mu, sd, 100)
    np.testing.assert_allclose(
        pdf(a, x), norm(mu, sd).pdf((x - loc) / scale) / scale
    )
