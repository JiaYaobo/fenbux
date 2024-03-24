import numpy as np
import tensorflow_probability.substrates.jax.bijectors as tfb

from fenbux.bijector import evaluate, ildj, inverse, ladj, Reshape


def test_reshape_forward():
    bij = Reshape(out_shape=(2, 5), in_shape=(10,))
    tf_bij = tfb.Reshape(event_shape_out=(2, 5), event_shape_in=(10,))
    x = np.random.uniform(size=(10,))
    y = evaluate(bij, x)
    y_tf = tf_bij.forward(x)
    assert np.allclose(y, y_tf)


def test_reshape_inverse():
    bij = Reshape(out_shape=(2, 5), in_shape=(10,))
    tf_bij = tfb.Reshape(event_shape_out=(2, 5), event_shape_in=(10,))
    x = np.random.uniform(size=(2, 5))
    y = evaluate(inverse(bij), x)
    y_tf = tf_bij.inverse(x)
    assert np.allclose(y, y_tf)


def test_reshape_ladj():
    bij = Reshape(out_shape=(2, 5), in_shape=(10,))
    tf_bij = tfb.Reshape(event_shape_out=(2, 5), event_shape_in=(10,))
    x = np.random.uniform(size=(10,))
    y = ladj(bij, x)
    y_tf = tf_bij.forward_log_det_jacobian(x, event_ndims=1)
    assert np.allclose(y, y_tf)


def test_reshape_ildj():
    bij = Reshape(out_shape=(2, 5), in_shape=(10,))
    tf_bij = tfb.Reshape(event_shape_out=(2, 5), event_shape_in=(10,))
    x = np.random.uniform(size=(2, 5))
    y = ildj(bij, x)
    y_tf = tf_bij.inverse_log_det_jacobian(x)
    assert np.allclose(y, y_tf)
