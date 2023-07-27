# Fenbu(分布)-X: A Simple Statistical Distribution Library in JAX

*fenbu* (pronounce like: /fen'bu:/)-X is a simple statistical distribution library in JAX. The library is encouraged by *Distribution.jl*, a wonderful julia library. In fenbux, We provide you:

* A simple and easy-to-use interface like **Distribution.jl**
* PyTree input/output
* Multiple dispatch
* All jax feautures (vmap, pmap, jit, autograd etc.)

## Examples

* Extract Attributes of Distributions 🤔

```python
import jax.numpy as jnp
from fenbux import Normal, variance, skewness, mean

μ = {'a': jnp.array([1., 2., 3.]), 'b': jnp.array([4., 5., 6.])} 
σ = {'a': jnp.array([4., 5., 6.]), 'b': jnp.array([7., 8., 9.])}

dist = Normal(μ, σ)
mean(dist) # {'a': Array([1., 2., 3.], dtype=float32), 'b': Array([4., 5., 6.], dtype=float32)}
variance(dist) # {'a': Array([16., 25., 36.], dtype=float32), 'b': Array([49., 64., 81.], dtype=float32)}
skewness(dist) # {'a': Array([0., 0., 0.], dtype=float32), 'b': Array([0., 0., 0.], dtype=float32)}
```

* Random Variables Generation

```python
import jax.random as jr
from fenbux import Normal, rand

key =  jr.PRNGKey(0)
x = {'a': {'c': {'d': {'e': 1.}}}}
y = {'a': {'c': {'d': {'e': 1.}}}}

dist = Normal(x, y)
rand(dist, key, shape=(3, )) # {'a': {'c': {'d': {'e': Array([1.6248107 , 0.69599575, 0.10169095], dtype=float32)}}}}
```

* Functions of Distribution

```python
import jax.numpy as jnp
from fenbux import Normal, cdf

μ = {'a': jnp.array([1., 2., 3.]), 'b': jnp.array([4., 5., 6.])}
σ = {'a': jnp.array([4., 5., 6.]), 'b': jnp.array([7., 8., 9.])}

dist = Normal(μ, σ)
cdf(dist, {'a': jnp.zeros((3, )), 'b': jnp.ones((3, ))})
# {'a': {'a': Array([0.4012937 , 0.34457827, 0.30853754], dtype=float32),
# 'b': Array([0.5       , 0.4207403 , 0.36944133], dtype=float32)},
#  'b': {'a': Array([0.28385457, 0.26598552, 0.25249255], dtype=float32),
#   'b': Array([0.33411756, 0.30853754, 0.28925735], dtype=float32)}}
```

* Compatible with JAX transformations 😃

```python
import jax.numpy as jnp
from jax import jit, vmap
from fenbux import Normal, logpdf

dist = Normal(0, jnp.ones((3, ))
# claim *use_batch=True* to use vmap
vmap(jit(logpdf), in_axes=(Normal(None, 0, use_batch=True), 0))(dist, jnp.zeros((3, )))
```

* Speed 🔦
  
```python
import jax.numpy as jnp
from scipy.stats import norm
from jax import jit
from fenbux import Normal, logpdf
from tensorflow_probability.substrates.jax.distributions import Normal as Normal2

dist = Normal(0, 1)
dist2 = Normal2(0, 1)
dist3 = norm(0, 1)
x = jnp.linspace(-5, 5, 100000)

%timeit jit(logpdf)(dist, x).block_until_ready()
%timeit jit(dist2.log_prob)(x).block_until_ready()
%timeit dist3.logpdf(x)
```

```
150 µs ± 1.29 µs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
10.5 ms ± 25.5 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
1.23 ms ± 33.7 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
```

## Installation

```bash
git clone https://github.com/JiaYaobo/fenbux.git
pip install -e .
```