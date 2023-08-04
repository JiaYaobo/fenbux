from jax._src.random import DTypeLikeFloat, DTypeLikeInt, KeyArray, Shape
from jaxtyping import ArrayLike, PyTree


PyTreeVar = PyTree[ArrayLike]

PyTreeKey = PyTree[KeyArray]

KeyArray = KeyArray

Shape = Shape

DTypeLikeFloat = DTypeLikeFloat

DTypeLikeInt = DTypeLikeInt
