import numpy as np
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd

from fenbux import logpdf
from fenbux.bijector import Exp, transform
from fenbux.univariate import Normal


def test_exp_logpdf():
    bij = Exp()
    tf_bij = tfb.Exp()
    dist = Normal(0.0, 1.0)
    x = np.random.uniform(size=(10,))
    y = logpdf(transform(dist, bij), x)
    y_tf = tfd.TransformedDistribution(tfd.Normal(0.0, 1.0), tf_bij).log_prob(x)
    assert np.allclose(y, y_tf)
