import jax.random as jr
import numpy as np
import pytest

from fenbux.core import (
    cdf,
    logcdf,
    logpdf,
    pdf,
    quantile,
    sf,
)
from fenbux.scipy_stats import cauchy
from fenbux.univariate import Cauchy


@pytest.mark.parametrize("loc, scale", [(0.0, 1.0), (1.0, 2.0), (-1.0, 0.5)])
def test_logpdf(loc: float, scale: float):
    x = np.random.standard_cauchy(10000) * scale + loc
    dist = Cauchy(loc, scale)
    np.testing.assert_allclose(logpdf(dist, x), cauchy(loc, scale).logpdf(x))
    

@pytest.mark.parametrize("loc, scale", [(0.0, 1.0), (1.0, 2.0), (-1.0, 0.5)])
def test_pdf(loc: float, scale: float):
    x = np.random.standard_cauchy(10000) * scale + loc
    dist = Cauchy(loc, scale)
    np.testing.assert_allclose(pdf(dist, x), cauchy(loc, scale).pdf(x))
    
    
@pytest.mark.parametrize("loc, scale", [(0.0, 1.0), (1.0, 2.0), (-1.0, 0.5)])
def test_cdf(loc: float, scale: float):
    x = np.random.standard_cauchy(10000) * scale + loc
    dist = Cauchy(loc, scale)
    np.testing.assert_allclose(cdf(dist, x), cauchy(loc, scale).cdf(x))
    

@pytest.mark.parametrize("loc, scale", [(0.0, 1.0), (1.0, 2.0), (-1.0, 0.5)])
def test_logcdf(loc: float, scale: float):
    x = np.random.standard_cauchy(10000) * scale + loc
    dist = Cauchy(loc, scale)
    np.testing.assert_allclose(logcdf(dist, x), cauchy(loc, scale).logcdf(x))
    

@pytest.mark.parametrize("loc, scale", [(0.0, 1.0), (1.0, 2.0), (-1.0, 0.5)])
def test_sf(loc: float, scale: float):
    x = np.random.standard_cauchy(10000) * scale + loc
    dist = Cauchy(loc, scale)
    np.testing.assert_allclose(sf(dist, x), cauchy(loc, scale).sf(x))
    

@pytest.mark.parametrize("loc, scale", [(0.0, 1.0), (1.0, 2.0), (-1.0, 0.5)])
def test_ppf(loc: float, scale: float):
    p = np.random.uniform(0.01, 0.99, 10000)
    dist = Cauchy(loc, scale)
    np.testing.assert_allclose(quantile(dist, p), cauchy(loc, scale).ppf(p))