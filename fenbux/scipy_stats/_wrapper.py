from typing import Any

from scipy import stats

from ..tree_utils import tree_map, tree_structures_all_eq


_dists = [
    "norm",
    "lognorm",
    "gamma",
    "geom",
    "chi2",
    "dirichlet",
    "poisson",
    "binom",
    "bernoulli",
    "uniform",
    "t",
    "f",
    "beta",
    "expon",
    "multivariate_normal",
    "weibull_min",
    "pareto",
    "invgauss",
    "wald",
    "logistic",
]
_methods = [
    "logcdf",
    "logpdf",
    "pdf",
    "cdf",
    "ppf",
    "pmf",
    "logpmf",
    "mean",
    "var",
    "std",
    "rvs",
    "skew",
    "kurtosis",
    "entropy",
    "sf",
    "stats",
    "isf",
]


class _ScipyDistWrapper(object):
    def __init__(self, dist: "str", *args, flat_kwargnames=None, **kwargs):
        if dist not in _dists:
            raise ValueError(f"dist must be one of {_dists}, got {dist}")

        if not tree_structures_all_eq(*args, **kwargs):
            raise ValueError("args and kwargs must have the same tree structure")
        scipy_dist = getattr(stats, dist)
        self.dist_tree = tree_map(
            lambda *_args, **_kwargs: scipy_dist(*_args, **_kwargs),
            *args,
            **kwargs,
            flat_kwargnames=flat_kwargnames,
        )

    def __call__(self, method: "str", *args, flat_kwargnames=None, **kwargs):
        if method not in _methods:
            raise ValueError(f"method must be one of {_methods}, got {method}")
        function_tree = tree_map(lambda d: getattr(d, method), self.dist_tree)
        return tree_map(
            lambda f,: tree_map(
                lambda *_args, **_kwargs: f(*_args, **_kwargs),
                *args,
                **kwargs,
                flat_kwargnames=flat_kwargnames,
            ),
            function_tree,
        )

    def mean(self):
        return self("mean")

    def var(self):
        return self("var")

    def std(self):
        return self("std")

    def skew(self):
        return self("skew")

    def kurtosis(self):
        return self("kurtosis")

    def entropy(self):
        return self("entropy")

    def logpdf(self, *args, **kwargs):
        return self("logpdf", *args, **kwargs)

    def pdf(self, *args, **kwargs):
        return self("pdf", *args, **kwargs)

    def logpmf(self, *args, **kwargs):
        return self("logpmf", *args, **kwargs)

    def pmf(self, *args, **kwargs):
        return self("pmf", *args, **kwargs)

    def logcdf(self, *args, **kwargs):
        return self("logcdf", *args, **kwargs)

    def cdf(self, *args, **kwargs):
        return self("cdf", *args, **kwargs)

    def ppf(self, *args, **kwargs):
        return self("ppf", *args, **kwargs)

    def sf(self, *args, **kwargs):
        return self("sf", *args, **kwargs)

    def isf(self, *args, **kwargs):
        return self("isf", *args, **kwargs)

    def rvs(self, *args, **kwargs):
        return self("rvs", *args, **kwargs)

    def stats(self, *args, **kwargs):
        return self("stats", *args, **kwargs)


class _ScipyDist(object):
    def __init__(self, dist: str) -> None:
        self.dist = dist

    def __call__(
        self, *args: Any, flat_kwargnames=None, **kwds: Any
    ) -> _ScipyDistWrapper:
        return _ScipyDistWrapper(
            self.dist, *args, flat_kwargnames=flat_kwargnames, **kwds
        )
