from jaxtyping import ArrayLike

from ..core import AbstractDistribution
from ..core._abstract_impls import _fenbux_dispatch
from ._typing import Bijector


@_fenbux_dispatch.abstract
def transform(d: AbstractDistribution, bijector: Bijector):
    """Transformed distribution
    Args:
        d (AbstractDistribution): A distribution object.
        bijector (Bijector): A bijector object.

    Example:
    >>> from fenbux.univariate import Normal
    >>> from fenbux.bijector import transform, Identity
    >>> dist = Normal(0.0, 1.0)
    >>> transformed(dist, Identity())
    """
    ...


@_fenbux_dispatch.abstract
def evaluate(bijector: Bijector, x: ArrayLike):
    """Evaluate a bijector
    Args:
        bijector (Bijector): A bijector object.
        x (ArrayLike): Value to transform.

    Example:
    >>> from fenbux.bijector import evaluate, Identity
    >>> evaluate(Identity(), 0.0)
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
def ladj(bijector: Bijector, x: ArrayLike):
    """Log absolute determinant of the jacobian of a bijector
    Args:
        bijector (Bijector): A bijector object.
        x (ArrayLike): Value to transform.

    Example:
    >>> from fenbux.bijector import ladj, Identity
    >>> ladj(Identity(), 0.0)
    """
    ...


@_fenbux_dispatch.abstract
def ildj(bijector: Bijector, x: ArrayLike):
    """Inverse log determinant of the jacobian of a bijector
    Args:
        bijector (Bijector): A bijector object.
        x (ArrayLike) : Value to transform.

    Example:
    >>> from fenbux.bijector import ildj, Identity
    >>> ildj(Identity(), 0.0)
    """
    ...


@_fenbux_dispatch.abstract
def value_and_ladj(bijector: Bijector, x):
    """Value and log absolute determinant of the jacobian of a bijector
    Args:
        bijector (Bijector): A bijector object.
        x (ArrayLike): Value to transform.

    Example:
    >>> from fenbux.bijector import value_and_ladj, Identity
    >>> value_and_ladj(Identity(), 0.0)
    """
    ...


@ildj.dispatch
def _ildj(b: Bijector, x: ArrayLike):
    return ladj(inverse(b), x)
