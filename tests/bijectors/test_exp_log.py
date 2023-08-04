import numpy as np
import tensorflow_probability.substrates.jax.bijectors as tfb

from fenbux import inverse, log_abs_det_jacobian, transform, value_and_ladj
from fenbux.bijectors import Exp, Log


def test_exp_transform():
    x = np.random.uniform(size=(1000,))
    assert np.allclose(transform(Exp(), x), tfb.Exp().forward(x))


def test_exp_inverse():
    x = np.random.gamma(shape=1.0, size=(1000,))
    assert np.allclose(transform(inverse(Exp()), x), tfb.Exp().inverse(x))


def test_exp_log_abs_det_jacobian():
    x = np.random.uniform(size=(1000,))
    assert np.allclose(
        log_abs_det_jacobian(Exp(), x),
        tfb.Exp().forward_log_det_jacobian(x, event_ndims=0),
    )


def test_exp_ildj():
    x = np.random.gamma(shape=1.0, size=(1000,))
    assert np.allclose(
        log_abs_det_jacobian(inverse(Exp()), x),
        tfb.Exp().inverse_log_det_jacobian(x, event_ndims=0),
    )


def test_log_transform():
    x = np.random.uniform(size=(1000,))
    assert np.allclose(transform(Log(), x), tfb.Log().forward(x))


def test_log_inverse():
    x = np.random.uniform(size=(1000,))
    assert np.allclose(transform(inverse(Log()), x), tfb.Log().inverse(x))


def test_log_abs_det_jacobian():
    x = np.random.uniform(size=(1000,))
    assert np.allclose(
        log_abs_det_jacobian(Log(), x),
        tfb.Log().forward_log_det_jacobian(x, event_ndims=0),
    )


def test_ildj():
    x = np.random.uniform(size=(1000,))
    assert np.allclose(
        log_abs_det_jacobian(inverse(Log()), x),
        tfb.Log().inverse_log_det_jacobian(x, event_ndims=0),
    )
