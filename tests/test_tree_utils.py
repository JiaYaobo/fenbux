import numpy as np

import fenbux.tree_utils as fbtu


def f(x, y, z):
    return x + y + z


def g(a, b):
    return a * b


def test_broadcast_pytree_arrays_shape():
    arr1 = np.ones((2, 3))
    arr2 = np.ones(())

    assert fbtu.broadcast_pytree_arrays_shapes(arr1, arr2).shape == (2, 3)


def test_broadcast_pytree_arrays():
    arr1 = np.ones((2, 3))
    arr2 = np.ones(())

    assert fbtu.broadcast_pytree_arrays(arr1, arr2)[0].shape == (2, 3)
    assert fbtu.broadcast_pytree_arrays(arr1, arr2)[1].shape == (2, 3)


def test_tree_structures_all_eq():
    assert fbtu.tree_structures_all_eq(1, 2, 3)
    assert fbtu.tree_structures_all_eq((1, 2), (3, 4), (5, 6))
    assert not fbtu.tree_structures_all_eq((1, 2), (3, 4), (5, 6, 7))


def test_tree_map_with_kwargs():
    assert fbtu.tree_map(f, 1, 2, z=3) == 6
    assert fbtu.tree_map(f, (1, 2), (3, 4), z=(5, 6)) == (9, 12)


def test_tree_map_with_flat_kwargs():
    assert fbtu.tree_map(f, (1, 2), (3, 4), z=3, flat_kwargnames=("z",)) == (7, 9)
    assert fbtu.tree_map(f, (1, 2), y=2, z=3, flat_kwargnames=("z", "y")) == (6, 7)
    assert fbtu.tree_map(f, x=1, y=2, z=3, flat_kwargnames=("x", "y", "z")) == 6


def test_tree_map_of_tree_map():
    assert fbtu.tree_map(
        g, (1, 2), fbtu.tree_map(f, (1, 2), (3, 4), z=3, flat_kwargnames=("z",))
    ) == (7, 18)
    assert fbtu.tree_map(
        g, (1, 2), b=fbtu.tree_map(f, (1, 2), (3, 4), z=3, flat_kwargnames=("z",))
    ) == (7, 18)
    assert fbtu.tree_map(
        g, a=(1, 2), b=fbtu.tree_map(f, (1, 2), (3, 4), z=3, flat_kwargnames=("z",))
    ) == (7, 18)
    assert fbtu.tree_map(
        g,
        a=1,
        b=fbtu.tree_map(f, (1, 2), (3, 4), z=3, flat_kwargnames=("z",)),
        flat_kwargnames=("a"),
    ) == (7, 9)
