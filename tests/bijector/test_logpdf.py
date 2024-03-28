import numpy as np
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd

from fenbux import logpdf
from fenbux.bijector import Chain, Exp, Scale, Shift, Tanh, transform
from fenbux.univariate import Normal


def test_exp_logpdf():
    bij = Exp()
    tf_bij = tfb.Exp()
    dist = Normal(0.0, 1.0)
    x = np.random.uniform(size=(10,))
    y = logpdf(transform(dist, bij), x)
    y_tf = tfd.TransformedDistribution(tfd.Normal(0.0, 1.0), tf_bij).log_prob(x)
    assert np.allclose(y, y_tf)


def test_scale_logpdf():
    bij = Scale(scale=2.0)
    tf_bij = tfb.Scale(scale=2.0)
    dist = Normal(0.0, 1.0)
    x = np.random.uniform(size=(10,))
    y = logpdf(transform(dist, bij), x)
    y_tf = tfd.TransformedDistribution(tfd.Normal(0.0, 1.0), tf_bij).log_prob(x)
    assert np.allclose(y, y_tf)


def test_shift_logpdf():
    bij = Shift(shift=2.0)
    tf_bij = tfb.Shift(shift=2.0)
    dist = Normal(0.0, 1.0)
    x = np.random.uniform(size=(10,))
    y = logpdf(transform(dist, bij), x)
    y_tf = tfd.TransformedDistribution(tfd.Normal(0.0, 1.0), tf_bij).log_prob(x)
    assert np.allclose(y, y_tf)


def test_chain_logpdf():
    bij = Chain([Exp(), Shift(shift=2.0)])
    tf_bij = tfb.Chain([tfb.Exp(), tfb.Shift(shift=2.0)])
    dist = Normal(0.0, 1.0)
    x = np.random.uniform(size=(10,))
    y = logpdf(transform(dist, bij), x)
    y_tf = tfd.TransformedDistribution(tfd.Normal(0.0, 1.0), tf_bij).log_prob(x)
    assert np.allclose(y, y_tf)
    

def test_tanh_logpdf():
    bij = Tanh()
    tf_bij = tfb.Tanh()
    dist = Normal(0.0, 1.0)
    x = np.random.uniform(size=(10,))
    y = logpdf(transform(dist, bij), x)
    y_tf = tfd.TransformedDistribution(tfd.Normal(0.0, 1.0), tf_bij).log_prob(x)
    assert np.allclose(y, y_tf)
