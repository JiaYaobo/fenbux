import numpy as np
import tensorflow_probability.substrates.jax.bijectors as tfb

from fenbux.bijector import (
    Chain,
    evaluate,
    Exp,
    ildj,
    inverse,
    ladj,
    Log,
)


def test_chain_forward():
    bij = Chain([Exp(), Log()])
    tf_bij = tfb.Chain([tfb.Exp(), tfb.Log()])
    x = np.random.uniform(1.0, 2.0, size=(10,))
    y = evaluate(bij, x)
    y_tf = tf_bij.forward(x)
    assert np.allclose(y, y_tf)


def test_chain_inverse():
    bij = Chain([Exp(), Log()])
    tf_bij = tfb.Chain([tfb.Exp(), tfb.Log()])
    x = np.random.uniform(1.0, 2.0, size=(10,))
    y = evaluate(inverse(bij), x)
    y_tf = tf_bij.inverse(x)
    assert np.allclose(y, y_tf)


def test_chain_ladj():
    bij = Chain([Exp(), Log()])
    tf_bij = tfb.Chain([tfb.Exp(), tfb.Log()])
    x = np.random.uniform(1.0, 2.0, size=(10,))
    y = ladj(bij, x)
    y_tf = tf_bij.forward_log_det_jacobian(x, event_ndims=0)
    assert np.allclose(y, y_tf)


def test_chain_ildj():
    bij = Chain([Exp(), Log()])
    tf_bij = tfb.Chain([tfb.Exp(), tfb.Log()])
    x = np.random.uniform(1.0, 2.0, size=(10,))
    y = ildj(bij, x)
    y_tf = tf_bij.inverse_log_det_jacobian(x)
    assert np.allclose(y, y_tf)
