import numpy as np
import tensorflow_probability.substrates.jax.bijectors as tfb

from fenbux import inverse, log_abs_det_jacobian, transform, value_and_ladj
from fenbux.bijectors import Reshape


def test_reshape_transform():
    x = np.random.uniform(size=(1000,))
    in_shape = (1000,)
    out_shape = (100, 10)
    assert np.allclose(
        transform(Reshape(in_shape, out_shape), x),
        tfb.Reshape(out_shape, in_shape).forward(x),
    )


def test_reshape_inverse():
    x = np.random.uniform(size=(100, 10))
    in_shape = (1000,)
    out_shape = (100, 10)
    assert np.allclose(
        transform(inverse(Reshape(in_shape, out_shape)), x),
        tfb.Reshape(out_shape, in_shape).inverse(x),
    )


def test_reshape_log_abs_det_jacobian():
    x = np.random.uniform(size=(100, 10))
    in_shape = (1000,)
    out_shape = (100, 10)
    assert np.allclose(
        log_abs_det_jacobian(Reshape(in_shape, out_shape), x),
        tfb.Reshape(out_shape, in_shape).forward_log_det_jacobian(x, event_ndims=1),
    )


def test_reshape_ildj():
    x = np.random.uniform(size=(100, 10))
    in_shape = (1000,)
    out_shape = (100, 10)
    assert np.allclose(
        log_abs_det_jacobian(inverse(Reshape(in_shape, out_shape)), x),
        tfb.Reshape(out_shape, in_shape).inverse_log_det_jacobian(x, event_ndims=2),
    )
