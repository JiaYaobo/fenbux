from functools import partial
from typing import Optional

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import jit, lax
from jax._src import core, dtypes
from jax._src.numpy.util import promote_dtypes_inexact
from jax._src.random import (
    _check_prng_key,
    _check_shape,
    _isnan,
    DTypeLikeInt,
    KeyArray,
    RealArray,
    split,
    uniform,
)
from jax._src.typing import Array, Shape


def _stirling_approx_tail(k):
    stirling_tail_vals = jnp.array(
        [
            0.0810614667953272,
            0.0413406959554092,
            0.0276779256849983,
            0.02079067210376509,
            0.0166446911898211,
            0.0138761288230707,
            0.0118967099458917,
            0.0104112652619720,
            0.00925546218271273,
            0.00833056343336287,
        ],
        dtype=k.dtype,
    )
    use_tail_values = k <= 9
    k = lax.clamp(0.0, k, 9.0)
    kp1sq = (k + 1) * (k + 1)
    approx = (1.0 / 12 - (1.0 / 360 - 1.0 / 1260 / kp1sq) / kp1sq) / (k + 1)
    k = jnp.floor(k)
    return lax.select(use_tail_values, stirling_tail_vals[jnp.int32(k)], approx)


@partial(jit, static_argnums=(3, 4, 5))
def _binomial_inversion(key, count, prob, shape, dtype, max_iters):
    log1minusprob = jnp.log1p(-prob)

    def body_fn(carry):
        i, num_geom, geom_sum, key = carry
        subkey, key = split(key)
        num_geom_out = lax.select(geom_sum <= count, num_geom + 1, num_geom)
        u = uniform(subkey, shape, prob.dtype)
        geom = jnp.ceil(jnp.log(u) / log1minusprob)
        geom_sum = geom_sum + geom
        return i + 1, num_geom_out, geom_sum, key

    def cond_fn(carry):
        i, geom_sum = carry[0], carry[2]
        return (geom_sum <= count).any() & (i < max_iters)

    num_geom_init = lax.full_like(prob, 0, prob.dtype, shape)
    geom_sum_init = lax.full_like(prob, 0, prob.dtype, shape)
    carry = (0, num_geom_init, geom_sum_init, key)
    k = lax.while_loop(cond_fn, body_fn, carry)[1]
    return (k - 1).astype(dtype)


@partial(jit, static_argnums=(3, 4, 5))
def _btrs(key, count, prob, shape, dtype, max_iters):
    # transforman-rejection algorithm
    # https://www.tandfonline.com/doi/abs/10.1080/00949659308811496
    stddev = jnp.sqrt(count * prob * (1 - prob))
    b = 1.15 + 2.53 * stddev
    a = -0.0873 + 0.0248 * b + 0.01 * prob
    c = count * prob + 0.5
    v_r = 0.92 - 4.2 / b
    r = prob / (1 - prob)
    alpha = (2.83 + 5.1 / b) * stddev
    m = jnp.floor((count + 1) * prob)

    def body_fn(carry):
        i, k_out, accepted, key = carry
        key, subkey_0, subkey_1 = split(key, 3)
        u = uniform(subkey_0, shape, prob.dtype)
        v = uniform(subkey_1, shape, prob.dtype)
        u = u - 0.5
        us = 0.5 - jnp.abs(u)
        accept1 = (us >= 0.07) & (v <= v_r)
        k = jnp.floor((2 * a / us + b) * u + c)
        reject = (k < 0) | (k > count)
        v = jnp.log(v * alpha / (a / (us * us) + b))
        ub = (
            (m + 0.5) * jnp.log((m + 1) / (r * (count - m + 1)))
            + (count + 1) * jnp.log((count - m + 1) / (count - k + 1))
            + (k + 0.5) * jnp.log(r * (count - k + 1) / (k + 1))
            + _stirling_approx_tail(m)
            + _stirling_approx_tail(count - m)
            - _stirling_approx_tail(k)
            - _stirling_approx_tail(count - k)
        )
        accept2 = v <= ub
        accept = accept1 | (~reject & accept2)
        k_out = lax.select(accept, k, k_out)
        accepted |= accept
        return i + 1, k_out, accepted, key

    def cond_fn(carry):
        i, accepted = carry[0], carry[2]
        return (~accepted).any() & (i < max_iters)

    k_init = lax.full_like(prob, -1, prob.dtype, shape)
    carry = (0, k_init, jnp.full(shape, False, jnp.bool_), key)
    return lax.while_loop(cond_fn, body_fn, carry)[1].astype(dtype)


@partial(jit, static_argnums=(3, 4))
def _binomial(key, count, prob, shape, dtype) -> Array:
    # The implementation matches TensorFlow and TensorFlow Probability:
    # https://github.com/tensorflow/tensorflow/blob/v2.2.0-rc3/tensorflow/core/kernels/random_binomial_op.cc
    # and tensorflow_probability.substrates.jax.distributions.Binomial
    # For n * p < 10, we use the binomial inverse algorithm; otherwise, we use btrs.
    if shape is None:
        shape = jnp.broadcast_shapes(jnp.shape(count), jnp.shape(prob))
    else:
        _check_shape("binomial", shape, np.shape(count), np.shape(prob))
    (prob,) = promote_dtypes_inexact(prob)
    count = lax.convert_element_type(count, prob.dtype)
    count = jnp.broadcast_to(count, shape)
    prob = jnp.broadcast_to(prob, shape)
    p_lt_half = prob < 0.5
    q = lax.select(p_lt_half, prob, 1.0 - prob)
    count_nan_or_neg = _isnan(count) | (count < 0.0)
    count_inf = jnp.isinf(count)
    q_is_nan = _isnan(q)
    q_le_0 = q <= 0.0
    q = lax.select(q_is_nan | q_le_0, lax.full_like(q, 0.01), q)
    use_inversion = count_nan_or_neg | (count * q <= 10.0)

    # consistent with np.random.binomial behavior for float count input
    count = jnp.floor(count)

    count_inv = lax.select(use_inversion, count, lax.full_like(count, 0.0))
    count_btrs = lax.select(use_inversion, lax.full_like(count, 1e4), count)
    q_btrs = lax.select(use_inversion, lax.full_like(q, 0.5), q)
    max_iters = dtype.type(jnp.finfo(dtype).max)
    samples = lax.select(
        use_inversion,
        _binomial_inversion(key, count_inv, q, shape, dtype, max_iters),
        _btrs(key, count_btrs, q_btrs, shape, dtype, max_iters),
    )
    # ensure nan q always leads to nan output and nan or neg count leads to nan
    # as discussed in https://github.com/google/jax/pull/16134#pullrequestreview-1446642709
    invalid = q_le_0 | q_is_nan | count_nan_or_neg
    samples = lax.select(
        invalid,
        jnp.full_like(samples, jnp.nan, dtype),
        samples,
    )

    # +inf count leads to inf
    samples = lax.select(
        count_inf & (~invalid),
        jnp.full_like(samples, jnp.inf, dtype),
        samples,
    )

    samples = lax.select(
        p_lt_half | count_nan_or_neg | q_is_nan | count_inf,
        samples,
        count.astype(dtype) - samples,
    )
    return samples


def binomial(
    key: KeyArray,
    n: RealArray,
    p: RealArray,
    shape: Optional[Shape] = None,
    dtype: DTypeLikeInt = dtypes.float_,
) -> Array:
    r"""Sample binomial random values.
    The values are returned according to the probability mass function:
    .. math::
        f(k;n,p) = \binom{n}{k}p^k(1-p)^{n-k}
    on the domain :math:`0 < p < 1`, and where :math:`n` is a nonnegative integer
    representing the number of trials and :math:`p` is a float representing the
    probability of success of an individual trial.
    Args:
      key: a PRNG key used as the random key.
      n: a float or array of floats broadcast-compatible with ``shape``
        representing the number of trials.
      p: a float or array of floats broadcast-compatible with ``shape``
        representing the the probability of success of an individual trial.
      shape: optional, a tuple of nonnegative integers specifying the result
        shape. Must be broadcast-compatible with ``n`` and ``p``.
        The default (None) produces a result shape equal to ``np.broadcast(n, p).shape``.
      dtype: optional, a int dtype for the returned values (default float64 if
        jax_enable_x64 is true, otherwise float32).
    Returns:
      A random array with the specified dtype and with shape given by
      ``np.broadcast(n, p).shape``.
    """
    key, _ = _check_prng_key(key)
    if not dtypes.issubdtype(dtype, np.floating):
        raise ValueError(
            "dtype argument to `binomial` must be a float " f"dtype, got {dtype}"
        )
    dtype = dtypes.canonicalize_dtype(dtype)
    if shape is not None:
        shape = core.canonicalize_shape(shape)
    return _binomial(key, n, p, shape, dtype)


def betabinom(
    key: KeyArray,
    n: RealArray,
    a: RealArray,
    b: RealArray,
    shape=None,
    dtype=dtypes.float_,
) -> Array:
    r"""Sample beta-binomial random values.
    The values are returned according to the probability mass function:
    .. math::
        f(k;n,a,b) = \binom{n}{k}\frac{B(k+a,n-k+b)}{B(a,b)}
    where :math:`B(a,b)` is the beta function.
    Args:
      key: a PRNG key used as the random key.
      n: a float or array of floats broadcast-compatible with ``shape``
        representing the number of trials.
      a: a float or array of floats broadcast-compatible with ``shape``
        representing the first shape parameter of the beta distribution.
      b: a float or array of floats broadcast-compatible with ``shape``
        representing the second shape parameter of the beta distribution.
      shape: optional, a tuple of nonnegative integers specifying the result
        shape. Must be broadcast-compatible with ``n``, ``a``, and ``b``.
        The default (None) produces a result shape equal to ``np.broadcast(n, a, b).shape``.
      dtype: optional, a int dtype for the returned values (default float64 if
        jax_enable_x64 is true, otherwise float32).
    Returns:
      A random array with the specified dtype and with shape given by
      ``np.broadcast(n, a, b).shape``.
    """
    key, _ = _check_prng_key(key)
    if not dtypes.issubdtype(dtype, np.floating):
        raise ValueError(
            "dtype argument to `betabinom` must be a float " f"dtype, got {dtype}"
        )
    dtype = dtypes.canonicalize_dtype(dtype)
    if shape is not None:
        shape = core.canonicalize_shape(shape)
    return _betabinom(key, n, a, b, shape, dtype)


def _betabinom(key, n, a, b, shape, dtype):
    if shape is None:
        shape = jnp.broadcast_shapes(jnp.shape(n), jnp.shape(a), jnp.shape(b))
    else:
        _check_shape("betabinom", shape, np.shape(n), np.shape(a), np.shape(b))
    (n,) = promote_dtypes_inexact(n)
    a = lax.convert_element_type(a, n.dtype)
    b = lax.convert_element_type(b, n.dtype)
    a = jnp.broadcast_to(a, shape)
    b = jnp.broadcast_to(b, shape)
    n = jnp.broadcast_to(n, shape)

    p = jr.beta(key, a, b, shape, dtype)
    return binomial(key, n, p, shape, dtype)
