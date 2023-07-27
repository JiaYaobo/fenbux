import jax.numpy as jnp
from jax import Array, lax
from jax.scipy.special import betainc
from tensorflow_probability.substrates.jax.math import special as tfp_special


# TODO (jiayaobo): rewrite jax version betaincinv...


def bdtr(k, n, p) -> Array:
    # TODO(jiayaobo): more precise version...
    ones = jnp.ones_like(n - k)
    safe_dn = jnp.where(lax.bitwise_or(k < 0, k >= n), ones, n - k)
    dk = betainc(a=safe_dn, b=k + 1, x=1 - p)
    return dk


def bdtri(k, n, y) -> Array:
    fk = jnp.floor(k)
    safe_dn = jnp.where(lax.bitwise_or(fk < 0, fk >= n), jnp.ones_like(fk), n - fk)
    _is_fk_0 = fk == 0
    _is_y_high = y > 0.8
    # fk == 0
    p1 = lax.select(
        _is_y_high,
        -jnp.expm1(jnp.log1p(y - 1.0) / safe_dn),
        1.0 - jnp.float_power(y, 1.0 / safe_dn),
    )
    # fk != 0
    dk = fk + 1
    p_ = betainc(a=safe_dn, b=dk, x=0.5)
    _is_p_high = p_ > 0.5
    p2 = lax.select(
        _is_p_high,
        tfp_special.betaincinv(dk, safe_dn, 1.0 - y),
        1.0 - tfp_special.betaincinv(safe_dn, dk, y),
    )
    p = lax.select(_is_fk_0, p1, p2)
    return p


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
