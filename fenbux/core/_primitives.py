from functools import wraps
from typing import Callable

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from ._abstract_impls import (
    _affine_impl,
    _cdf_impl,
    _cf_impl,
    _entropy_impl,
    _kurtosis_impl,
    _logcdf_impl,
    _logpdf_impl,
    _logpmf_impl,
    _mean_impl,
    _mgf_impl,
    _params_impl,
    _pdf_impl,
    _pmf_impl,
    _quantile_impl,
    _rand_impl,
    _sf_impl,
    _skewness_impl,
    _standard_dev_impl,
    _support_impl,
    _variance_impl,
)


def _to_struct(x):
    if isinstance(x, jax.core.ShapedArray):
        return jax.ShapeDtypeStruct(x.shape, x.dtype)
    elif isinstance(x, jax.core.AbstractValue):
        raise NotImplementedError(
            "Only supports working with JAX arrays; not "
            f"other abstract values. Got abstract value {x}."
        )
    else:
        return x


def _to_shapedarray(x):
    if isinstance(x, jax.ShapeDtypeStruct):
        return jax.core.ShapedArray(x.shape, x.dtype)
    else:
        return x


def create_prim_at_val(name, impl) -> Callable:
    @eqxi.filter_primitive_def
    def _abstract_eval(dist, x):
        dist, x = jtu.tree_map(_to_struct, (dist, x))
        out = eqx.filter_eval_shape(impl, dist, x)
        out = jtu.tree_map(_to_shapedarray, out)
        return out

    prim = eqxi.create_vprim(
        name,
        eqxi.filter_primitive_def(impl),
        _abstract_eval,
        None,
        None,
    )
    prim.def_impl(eqxi.filter_primitive_def(impl))
    eqxi.register_impl_finalisation(prim)

    @jax.custom_jvp
    @eqx.filter_jit
    def custom_prim(dist, x):
        return eqxi.filter_primitive_bind(prim, dist, x)

    @custom_prim.defjvp
    def impl_jvp(primals, tangents):
        return jax.jvp(impl, primals, tangents)

    @wraps(impl)
    def prim_fun(dist, x):
        return custom_prim(dist, x)

    return prim_fun


def create_prim_dist(name, impl) -> Callable:
    @eqxi.filter_primitive_def
    def _abstract_eval(dist):
        dist = jtu.tree_map(_to_struct, dist)
        out = eqx.filter_eval_shape(impl, dist)
        out = jtu.tree_map(_to_shapedarray, out)
        return out

    prim = eqxi.create_vprim(
        name,
        eqxi.filter_primitive_def(impl),
        _abstract_eval,
        None,
        None,
    )
    prim.def_impl(eqxi.filter_primitive_def(impl))
    eqxi.register_impl_finalisation(prim)

    @jax.custom_jvp
    @eqx.filter_jit
    def custom_prim(dist):
        return eqxi.filter_primitive_bind(prim, dist)

    @custom_prim.defjvp
    def impl_jvp(primals, tangents):
        return jax.jvp(impl, primals, tangents)

    @wraps(impl)
    def prim_fun(dist):
        return custom_prim(dist)

    return prim_fun


def create_prim_at_vals_2_(name: str, impl: Callable) -> Callable:
    @eqxi.filter_primitive_def
    def _abstract_eval(dist, arg1, arg2):
        dist = jtu.tree_map(_to_struct, dist)
        arg1 = jtu.tree_map(_to_struct, arg1)
        arg2 = jtu.tree_map(_to_struct, arg2)
        out = eqx.filter_eval_shape(impl, dist, arg1, arg2)
        out = jtu.tree_map(_to_shapedarray, out)
        return out

    prim = eqxi.create_vprim(
        name,
        eqxi.filter_primitive_def(impl),
        _abstract_eval,
        None,
        None,
    )
    prim.def_impl(eqxi.filter_primitive_def(impl))
    eqxi.register_impl_finalisation(prim)

    @jax.custom_jvp
    @eqx.filter_jit
    def custom_jvp_prim(dist, arg1, arg2):
        return eqxi.filter_primitive_bind(prim, dist, arg1, arg2)

    @custom_jvp_prim.defjvp
    def impl_jvp(primals, tangents):
        return jax.jvp(impl, primals, tangents)

    @wraps(impl)
    def prim_fun(dist, arg1, arg2):
        return custom_jvp_prim(dist, arg1, arg2)

    return prim_fun


def create_prim_rand_(name, impl) -> Callable:
    @eqxi.filter_primitive_def
    def _abstract_eval(dist, key, shape, dtype=jnp.floating):
        dist = jtu.tree_map(_to_struct, dist)
        key = jtu.tree_map(_to_struct, key)
        shape = jtu.tree_map(_to_struct, shape)
        out = eqx.filter_eval_shape(impl, dist, key, shape, dtype)
        out = jtu.tree_map(_to_shapedarray, out)
        return out

    prim = eqxi.create_vprim(
        name,
        eqxi.filter_primitive_def(impl),
        _abstract_eval,
        None,
        None,
    )
    prim.def_impl(eqxi.filter_primitive_def(impl))
    eqxi.register_impl_finalisation(prim)

    @jax.custom_jvp
    @eqx.filter_jit
    def custom_jvp_prim(dist, key, shape, dtype):
        return eqxi.filter_primitive_bind(prim, dist, key, shape, dtype)

    @custom_jvp_prim.defjvp
    def impl_jvp(primals, tangents):
        return jax.jvp(impl, primals, tangents)

    @wraps(impl)
    def prim_fun(dist, key, shape, dtype=jnp.floating):
        return custom_jvp_prim(dist, key, shape, dtype)

    return prim_fun


logpdf = create_prim_at_val("logpdf", _logpdf_impl)
pdf = create_prim_at_val("pdf", _pdf_impl)
logcdf = create_prim_at_val("logcdf", _logcdf_impl)
cdf = create_prim_at_val("cdf", _cdf_impl)
sf = create_prim_at_val("sf", _sf_impl)
logpmf = create_prim_at_val("logpmf", _logpmf_impl)
pmf = create_prim_at_val("pmf", _pmf_impl)
mgf = create_prim_at_val("mgf", _mgf_impl)
cf = create_prim_at_val("cf", _cf_impl)
quantile = create_prim_at_val("quantile", _quantile_impl)

params = create_prim_dist("params", _params_impl)
support = create_prim_dist("support", _support_impl)
mean = create_prim_dist("mean", _mean_impl)
variance = create_prim_dist("variance", _variance_impl)
standard_dev = create_prim_dist("standard_dev", _standard_dev_impl)
skewness = create_prim_dist("skewness", _skewness_impl)
kurtosis = create_prim_dist("kurtosis", _kurtosis_impl)
entropy = create_prim_dist("entropy", _entropy_impl)

affine = create_prim_at_vals_2_("affine", _affine_impl)

rand = create_prim_rand_("rand", _rand_impl)