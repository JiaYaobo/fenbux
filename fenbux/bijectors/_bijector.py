### Map support(P) to R^d
from ..core import bijector
from ..univariate.continuous import Normal
from ._base import Identity


@bijector.dispatch
def _bijector(d: Normal):
    return Identity()
