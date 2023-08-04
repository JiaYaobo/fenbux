import numpy as np
import tensorflow_probability.substrates.jax.bijectors as tfb

from fenbux import inverse, log_abs_det_jacobian, transform, value_and_ladj
from fenbux.bijectors import Scale, Shift


def test_scale_transform():
    x = np.random.uniform(size=(1000,))
    scale = np.random.uniform(1.0, 2.0, size=(1000,))
    assert np.allclose(transform(Scale(scale), x), tfb.Scale(scale).forward(x))


def test_scale_inverse():
    x = np.random.uniform(size=(1000,))
    scale = np.random.uniform(1.0, 2.0, size=(1000,))
    assert np.allclose(transform(inverse(Scale(scale)), x), tfb.Scale(scale).inverse(x))


def test_scale_log_abs_det_jacobian():
    x = np.random.uniform(size=(1000,))
    scale = np.random.uniform(1.0, 2.0, size=(1000,))
    assert np.allclose(
        log_abs_det_jacobian(Scale(scale), x),
        tfb.Scale(scale).forward_log_det_jacobian(x, event_ndims=0),
    )


def test_scale_ildj():
    x = np.random.uniform(size=(1000,))
    scale = np.random.uniform(1.0, 2.0, size=(1000,))
    assert np.allclose(
        log_abs_det_jacobian(inverse(Scale(scale)), x),
        tfb.Scale(scale).inverse_log_det_jacobian(x, event_ndims=0),
    )


def test_shift_transform():
    x = np.random.uniform(size=(1000,))
    shift = np.random.uniform(-1.0, 1.0, size=(1000,))
    assert np.allclose(transform(Shift(shift), x), tfb.Shift(shift).forward(x))


def test_shift_inverse():
    x = np.random.uniform(size=(1000,))
    shift = np.random.uniform(-1.0, 1.0, size=(1000,))
    assert np.allclose(transform(inverse(Shift(shift)), x), tfb.Shift(shift).inverse(x))


def test_shift_log_abs_det_jacobian():
    x = np.random.uniform(size=(1000,))
    shift = np.random.uniform(-1.0, 1.0, size=(1000,))
    assert np.allclose(
        log_abs_det_jacobian(Shift(shift), x),
        tfb.Shift(shift).forward_log_det_jacobian(x, event_ndims=0),
    )


def test_shift_ildj():
    x = np.random.uniform(size=(1000,))
    shift = np.random.uniform(-1.0, 1.0, size=(1000,))
    assert np.allclose(
        log_abs_det_jacobian(inverse(Shift(shift)), x),
        tfb.Shift(shift).inverse_log_det_jacobian(x, event_ndims=0),
    )
