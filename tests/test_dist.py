import jax.numpy as jnp

from .helpers import FakeDistribution, shaped_allclose


dist = FakeDistribution(
    {"a": 1.0, "b": 2.0},
    {"a": jnp.ones((2,)), "b": jnp.ones((2,))},
    {"a": 1.0, "b": jnp.ones((10, 2))},
)


def test_broadcast_shapes():
    shape = dist.broadcast_shapes()
    assert shape["a"].shape == (2,)
    assert shape["b"].shape == (10, 2)


def test_broadcast_params():
    params = dist.broadcast_params()
    assert shaped_allclose(
        params.arg1, {"a": jnp.ones((2,)), "b": 2.0 * jnp.ones((10, 2))}
    )
    assert shaped_allclose(params.arg2, {"a": jnp.ones((2,)), "b": jnp.ones((10, 2))})
    assert shaped_allclose(params.arg3, {"a": jnp.ones((2,)), "b": jnp.ones((10, 2))})


def test_broadcast_to():
    shape = (10, 2)
    params = dist.broadcast_to(shape)
    assert shaped_allclose(
        params.arg1, {"a": jnp.ones((10, 2)), "b": 2.0 * jnp.ones((10, 2))}
    )
    assert shaped_allclose(
        params.arg2, {"a": jnp.ones((10, 2)), "b": jnp.ones((10, 2))}
    )
    assert shaped_allclose(
        params.arg3, {"a": jnp.ones((10, 2)), "b": jnp.ones((10, 2))}
    )
