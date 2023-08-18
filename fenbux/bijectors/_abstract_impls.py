from ..core import AbstractDistribution
from ..core._abstract_impls import _fenbux_dispatch
from ..core._typing import PyTree
from ._typing import Bijector


@_fenbux_dispatch.abstract
def transformed(d: AbstractDistribution, bijector: Bijector):
    """Transformed distribution
    Args:
        d (AbstractDistribution): A distribution object.
        bijector (PyTree): A bijector object.

    Example:
    >>> from fenbux import Normal, transformed, Identity
    >>> dist = Normal(0.0, 1.0)
    >>> transformed(dist, Identity())
    """
    ...


@_fenbux_dispatch.abstract
def bijector(d: AbstractDistribution):
    """Bijector of a distribution
    Args:
        d (AbstractDistribution): A distribution object.

    Example:
    >>> from fenbux import Normal, bijector
    >>> dist = Normal(0.0, 1.0)
    >>> bijector(dist)
    """
    ...


@_fenbux_dispatch.abstract
def transform(bijecto: Bijector, x: PyTree):
    """Transform a value using a bijector
    Args:
        bijector (Bijector): A bijector object.
        x (PyTree): Value to transform.

    Example:
    >>> from fenbux import transform, Identity
    >>> transform(Identity(), 0.0)

    """
    ...


@_fenbux_dispatch.abstract
def inverse(bijector: Bijector):
    """Inverse of a bijector
    Args:
        bijector (Bijector): A bijector object.

    Example:
    >>> from fenbux import inverse, Exp
    >>> inverse(Exp())

    """
    ...


@_fenbux_dispatch.abstract
def log_abs_det_jacobian(bijector: Bijector, x: PyTree):
    """Log absolute determinant of the jacobian of a bijector
    Args:
        bijector (Bijector): A bijector object.
        x (PyTree): Value to transform.

    Example:
    >>> from fenbux import log_abs_det_jacobian, Identity
    >>> log_abs_det_jacobian(Identity(), 0.0)
    """
    ...


@_fenbux_dispatch.abstract
def value_and_ladj(bijector: Bijector, x: PyTree):
    """Value and log absolute determinant of the jacobian of a bijector
    Args:
        bijector (Bijector): A bijector object.
        x (PyTree): Value to transform.

    Example:
    >>> from fenbux import value_and_ladj, Identity
    >>> value_and_ladj(Identity(), 0.0)
    """
    ...
