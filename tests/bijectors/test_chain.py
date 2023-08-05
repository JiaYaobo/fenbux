import numpy as np
import tensorflow_probability.substrates.jax.bijectors as tfb

from fenbux import inverse, log_abs_det_jacobian, transform
from fenbux.bijectors import Chain, Exp, Log, Scale, Shift


def test_chain_transform():
    x = np.random.uniform(1.0, 5.0, size=(1000,))
    assert np.allclose(
        transform(Chain([Exp(), Scale(2.0), Log(), Shift(1.0)]), x),
        tfb.Chain(
            bijectors=[tfb.Exp(), tfb.Scale(2.0), tfb.Log(), tfb.Shift(1.0)]
        ).forward(x),
    )


def test_chain_inverse():
    x = np.random.uniform(1.0, 5.0, size=(1000,)).astype(np.float32)
    assert np.allclose(
        transform(inverse(Chain([Exp(), Log(), Shift(1.0)])), x),
        tfb.Chain(bijectors=[tfb.Exp(), tfb.Log(), tfb.Shift(1.0)]).inverse(
            x, event_ndims=0
        ),
    )


def test_chain_log_abs_det_jacobian():
    x = np.random.uniform(1.0, 5.0, size=(1000,))
    assert np.allclose(
        log_abs_det_jacobian(Chain([Exp(), Scale(2.0), Log(), Shift(1.0)]), x),
        tfb.Chain(
            bijectors=[tfb.Exp(), tfb.Scale(2.0), tfb.Log(), tfb.Shift(1.0)]
        ).forward_log_det_jacobian(x, event_ndims=0),
    )


def test_chain_ildj():
    x = np.random.uniform(1.0, 5.0, size=(1000,))
    assert np.allclose(
        log_abs_det_jacobian(inverse(Chain([Exp(), Scale(2.0), Log(), Shift(1.0)])), x),
        tfb.Chain(
            bijectors=[tfb.Exp(), tfb.Scale(2.0), tfb.Log(), tfb.Shift(1.0)]
        ).inverse_log_det_jacobian(x, event_ndims=0),
    )
