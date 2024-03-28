import numpy as np
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd

from fenbux import logcdf
from fenbux.bijector import Chain, Exp, Scale, Shift, Tanh, transform
from fenbux.univariate import Normal


def test_exp_logcdf():
    bij = Exp()
    tf_bij = tfb.Exp()
    dist = Normal(0.0, 1.0)
    x = np.random.uniform(size=(10,))
    y = logcdf(transform(dist, bij), x)
    y_tf = tfd.TransformedDistribution(tfd.Normal(0.0, 1.0), tf_bij).log_cdf(x)
    assert np.allclose(y, y_tf)


def test_scale_logcdf():
    bij = Scale(scale=2.0)
    tf_bij = tfb.Scale(scale=2.0)
    dist = Normal(0.0, 1.0)
    x = np.random.uniform(size=(10,))
    y = logcdf(transform(dist, bij), x)
    y_tf = tfd.TransformedDistribution(tfd.Normal(0.0, 1.0), tf_bij).log_cdf(x)
    assert np.allclose(y, y_tf)


def test_shift_logcdf():
    bij = Shift(shift=2.0)
    tf_bij = tfb.Shift(shift=2.0)
    dist = Normal(0.0, 1.0)
    x = np.random.uniform(size=(10,))
    y = logcdf(transform(dist, bij), x)
    y_tf = tfd.TransformedDistribution(tfd.Normal(0.0, 1.0), tf_bij).log_cdf(x)
    assert np.allclose(y, y_tf)


def test_chain_logcdf():
    bij = Chain([Exp(), Shift(shift=2.0), Scale(scale=-2.0)])
    tf_bij = tfb.Chain([tfb.Exp(), tfb.Shift(shift=2.0), tfb.Scale(scale=-2.0)])
    dist = Normal(0.0, 1.0)
    x = np.random.uniform(size=(10,))
    y = logcdf(transform(dist, bij), x)
    y_tf = tfd.TransformedDistribution(tfd.Normal(0.0, 1.0), tf_bij).log_cdf(x)
    assert np.allclose(y, y_tf)
    
    
def test_tanh_logcdf():
    bij = Tanh()
    tf_bij = tfb.Tanh()
    dist = Normal(0.0, 1.0)
    x = np.random.uniform(size=(10,))
    y = logcdf(transform(dist, bij), x)
    y_tf = tfd.TransformedDistribution(tfd.Normal(0.0, 1.0), tf_bij).log_cdf(x)
    assert np.allclose(y, y_tf)
