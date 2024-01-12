from typing import Any, Protocol, Union

import numpy as np
from jax import Array
from jax._src.random import KeyArray, Shape
from jaxtyping import ArrayLike, PyTree


DType = np.dtype


class SupportsDType(Protocol):
    @property
    def dtype(self) -> DType:
        ...


# since https://github.com/google/jax/commit/911f745775137025af8c23a4ad4b84960a12e693,
# Any to type[Any] and SupportsDType is not bearable
# so we keep the original looser typing
DTypeLike = Union[Any, str, np.dtype, SupportsDType]

DTypeLikeInt = DTypeLike
DTypeLikeUInt = DTypeLike
DTypeLikeFloat = DTypeLike


PyTreeVar = PyTree[ArrayLike]

PyTreeKey = PyTree[Array]
