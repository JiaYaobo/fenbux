from typing import Union

from jax._src.random import PRNGKey, Shape
from jaxtyping import ArrayLike, Complex, Float, Int, PyTree


PyTreeVar = PyTree[
    Union[Float[ArrayLike, "..."], Int[ArrayLike, "..."], Complex[ArrayLike, "..."]]
]

PyTreeKey = PyTree[PRNGKey]

Shape = Shape