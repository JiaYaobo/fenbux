import numpy as np
import tensorflow_probability.substrates.jax.bijectors as tfb

from fenbux.bijector import evaluate, ildj, inverse, ladj, Scale


def test_scale_forward():
    bij = Scale(scale=2.0)
    tf_bij = tfb.Scale(scale=2.0)
    x = np.random.uniform(size=(10,))
    y = evaluate(bij, x)
    y_tf = tf_bij.forward(x)
    assert np.allclose(y, y_tf)
    

def test_scale_inverse():
    bij = Scale(scale=2.0)
    tf_bij = tfb.Scale(scale=2.0)
    x = np.random.uniform(size=(10,))
    y = evaluate(inverse(bij), x)
    y_tf = tf_bij.inverse(x)
    assert np.allclose(y, y_tf)


def test_scale_ladj():
    bij = Scale(scale=2.0)
    tf_bij = tfb.Scale(scale=2.0)
    x = np.random.uniform(size=(100,))
    y = ladj(bij, x)
    y_tf = tf_bij.forward_log_det_jacobian(x, event_ndims=0)
    assert np.allclose(y, y_tf)
    

def test_scale_ildj():
    bij = Scale(scale=2.0)
    tf_bij = tfb.Scale(scale=2.0)
    x = np.random.uniform(size=(10,))
    y = ildj(bij, x)
    y_tf = tf_bij.inverse_log_det_jacobian(x)
    assert np.allclose(y, y_tf)
    