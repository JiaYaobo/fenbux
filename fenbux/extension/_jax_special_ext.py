from jax import lax
from jax.scipy.special import betainc
from tensorflow_probability.substrates.jax.math import special as tfp_special


def fdtrc(a, b, x):
    dtype = lax.dtype(x)
    a = lax.convert_element_type(a, dtype)
    b = lax.convert_element_type(b, dtype)
    w = lax.div(b, lax.add(b, lax.mul(a, x)))
    half = lax.convert_element_type(0.5, w.dtype)
    a = lax.mul(half, a)
    b = lax.mul(half, b)
    return betainc(a, b, w)


def fdtr(a, b, x):
    dtype = lax.dtype(x)
    a = lax.convert_element_type(a, dtype)
    b = lax.convert_element_type(b, dtype)
    w = lax.mul(a, x)
    w = lax.div(w, lax.add(b, w))
    half = lax.convert_element_type(0.5, dtype)
    a = lax.mul(half, a)
    b = lax.mul(half, b)
    return betainc(a, b, w)


def fdtri(a, b, y):
    dtype = lax.dtype(y)
    one = lax.convert_element_type(1.0, dtype)
    half = lax.convert_element_type(0.5, dtype)
    eps = lax.convert_element_type(1e-3, dtype)
    a = lax.convert_element_type(a, dtype)
    b = lax.convert_element_type(b, dtype)
    y = lax.sub(one, y)
    a = lax.mul(half, a)
    b = lax.mul(half, b)
    w = betainc(a, b, half)
    cond0 = (w > y) | (y < eps)
    w = lax.select(
        cond0,
        tfp_special.betaincinv(b, a, y),
        tfp_special.betaincinv(a, b, lax.sub(one, y)),
    )
    left_out = lax.div(lax.sub(b, lax.mul(b, w)), lax.mul(a, w))
    right_out = lax.div(lax.mul(b, w), lax.mul(a, lax.sub(one, w)))
    x = lax.select(cond0, left_out, right_out)
    return x