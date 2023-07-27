import numpy as np
import pytest

from fenbux import affine, mean, Normal
from fenbux.scipy_stats import norm


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
