import numpy as np
from scipy import stats

from fenbux.scipy_stats import bernoulli, binom, chi2, gamma, norm, poisson, t, uniform

from .helpers import shaped_allclose


def test_create_dists():
    assert norm(1.0, 2.0).mean() == stats.norm(1.0, 2.0).mean()
    assert gamma(1.0, 2.0).mean() == stats.gamma(1.0, 2.0).mean()
    assert chi2(1.0).mean() == stats.chi2(1.0).mean()
    assert poisson(1.0).mean() == stats.poisson(1.0).mean()
    assert uniform(1.0, 2.0).mean() == stats.uniform(1.0, 2.0).mean()
    assert t(df=2.0).mean() == stats.t(df=2.0).mean()
    assert binom(1.0, 0.2).mean() == stats.binom(1.0, 0.2).mean()
    assert bernoulli(0.2).mean() == stats.bernoulli(0.2).mean()


def test_create_tree_dists():
    assert norm({"a": 1.0, "b": 2.0}, {"a": 2.0, "b": 3.0}).mean() == {
        "a": stats.norm(1.0, 1.0).mean(),
        "b": stats.norm(2.0, 2.0).mean(),
    }
    assert gamma({"a": 1.0, "b": 2.0}, {"a": 2.0, "b": 3.0}).mean() == {
        "a": stats.gamma(1.0, 2.0).mean(),
        "b": stats.gamma(2.0, 3.0).mean(),
    }
    assert chi2({"a": 1.0, "b": 2.0}).mean() == {
        "a": stats.chi2(1.0).mean(),
        "b": stats.chi2(2.0).mean(),
    }
    assert poisson({"a": 1.0, "b": 2.0}).mean() == {
        "a": stats.poisson(1.0).mean(),
        "b": stats.poisson(2.0).mean(),
    }


def test_tree_io():
    normal = norm({"a": 1.0, "b": 2.0}, {"a": 2.0, "b": 3.0})
    x = np.random.uniform(size=10000)
    assert shaped_allclose(
        normal.logpdf(x=x, flat_kwargnames=["x"]),
        {"a": stats.norm(1.0, 2.0).logpdf(x), "b": stats.norm(2.0, 3.0).logpdf(x)},
    )
    assert shaped_allclose(
        normal.pdf(x=x, flat_kwargnames=["x"]),
        {"a": stats.norm(1.0, 2.0).pdf(x), "b": stats.norm(2.0, 3.0).pdf(x)},
    )
    assert shaped_allclose(
        normal.cdf(x=x, flat_kwargnames=["x"]),
        {"a": stats.norm(1.0, 2.0).cdf(x), "b": stats.norm(2.0, 3.0).cdf(x)},
    )
    assert shaped_allclose(
        normal.ppf(q=x, flat_kwargnames=["q"]),
        {"a": stats.norm(1.0, 2.0).ppf(x), "b": stats.norm(2.0, 3.0).ppf(x)},
    )
