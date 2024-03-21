import numpy as np
import tensorflow_probability.substrates.jax.bijectors as tfb

from fenbux.bijector import evaluate, Exp, ildj, inverse, ladj, Log


def test_exp_forward():
    bij = Exp()
    tf_bij = tfb.Exp()
    x = np.random.uniform(size=(10,))
    y = evaluate(bij, x)
    y_tf = tf_bij.forward(x)
    assert np.allclose(y, y_tf)


def test_exp_inverse():
    bij = Exp()
    tf_bij = tfb.Exp()
    x = np.random.uniform(size=(10,))
    y = evaluate(inverse(bij), x)
    y_tf = tf_bij.inverse(x)
    assert np.allclose(y, y_tf)


def test_exp_ladj():
    bij = Exp()
    tf_bij = tfb.Exp()
    x = np.random.uniform(size=(100,))
    y = ladj(bij, x)
    y_tf = tf_bij.forward_log_det_jacobian(x, event_ndims=0)
    assert np.allclose(y, y_tf)


def test_exp_ildj():
    bij = Exp()
    tf_bij = tfb.Exp()
    x = np.random.uniform(size=(10,))
    y = ildj(bij, x)
    y_tf = tf_bij.inverse_log_det_jacobian(x)
    assert np.allclose(y, y_tf)


def test_log_forward():
    bij = Log()
    tf_bij = tfb.Log()
    x = np.random.uniform(1.0, 10.0, size=(10,))
    y = evaluate(bij, x)
    y_tf = tf_bij.forward(x)
    assert np.allclose(y, y_tf)
    
    
def test_log_inverse():
    bij = Log()
    tf_bij = tfb.Log()
    x = np.random.uniform(1.0, 10.0, size=(10,))
    y = evaluate(inverse(bij), x)
    y_tf = tf_bij.inverse(x)
    assert np.allclose(y, y_tf)
    


def test_log_ladj():
    bij = Log()
    tf_bij = tfb.Log()
    x = np.random.uniform(1.0, 10.0, size=(10,))
    y = ladj(bij, x)
    y_tf = tf_bij.forward_log_det_jacobian(x)
    assert np.allclose(y, y_tf)
    

def test_log_ildj():
    bij = Log()
    tf_bij = tfb.Log()
    x = np.random.uniform(1.0, 10.0, size=(1,))
    y = ildj(bij, x)
    y_tf = tf_bij.inverse_log_det_jacobian(x)
    assert np.allclose(y, y_tf)