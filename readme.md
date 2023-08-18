# FenbuX

*A Simple Probalistic Distribution Library in JAX*

*fenbu* (分布, pronounce like: /fen'bu:/)-X is a simple probalistic distribution library in JAX. The library is encouraged by *Distributions.jl* and *Bijectors.jl*. In fenbux, We provide you:

* A simple and easy-to-use interface like **Distributions.jl**
* PyTree input/output
* Multiple dispatch for different distributions based on [plum-dispatch](https://github.com/beartype/plum)
* Bijectors interface like **Bijectors.jl**
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

* Functions of Distribution 👩‍🎓

CDF, PDF, and more...

```python
import jax.numpy as jnp
from fenbux import Normal, cdf, logpdf

μ = jnp.array([1., 2., 3.])
σ = jnp.array([4., 5., 6.])

dist = Normal(μ, σ)
cdf(dist, jnp.array([1., 2., 3.])) # Array([0.5, 0.5, 0.5], dtype=float32)
logpdf(dist, jnp.array([1., 2., 3.])) # Array([-2.305233 , -2.5283763, -2.7106981], dtype=float32)
```

* Bijectors transformations 🤖
  
```python
import jax.numpy as jnp
from fenbux import Normal, logpdf, LogNormal
from fenbux.bijectors import Exp, transformed, Log

dist = Normal(0, 1)
dist2 = LogNormal(0, 1)
dist3 = transformed(dist2, Exp())
logpdf(dist, jnp.array([1., 2., 3.])), logpdf(dist3, jnp.array([1., 2., 3.]))
```

```
(Array([-1.4189385, -2.9189386, -5.4189386], dtype=float32),
 Array([-1.4189386, -2.9189386, -5.4189386], dtype=float32))
```

* Compatible with JAX transformations 😃

```python
import jax.numpy as jnp
from jax import jit, vmap
from fenbux import Normal, logpdf

dist = Normal(0, jnp.ones((3, )))
# set claim use_batch=True to use vmap
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
34.4 µs ± 678 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
9.64 ms ± 177 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
1.17 ms ± 51.2 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
```

## Installation

```bash
git clone https://github.com/JiaYaobo/fenbux.git
pip install -e .
```

## Reference

* [Distributions.jl](https://github.com/JuliaStats/Distributions.jl)
* [Bijectors.jl](https://github.com/TuringLang/Bijectors.jl)