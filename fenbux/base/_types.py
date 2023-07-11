from typing import Union

from jax._src.random import DTypeLikeFloat, DTypeLikeInt, KeyArray, Shape
from jaxtyping import ArrayLike, Complex, Float, Int, PyTree


PyTreeVar = PyTree[ArrayLike]

PyTreeKey = PyTree[KeyArray]

KeyArray = KeyArray

Shape = Shape

DTypeLikeFloat = DTypeLikeFloat

DTypeLikeInt = DTypeLikeInt
