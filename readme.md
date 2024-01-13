# FenbuX

*A Simple Probalistic Distribution Library in JAX*

*fenbu* (分布, pronounce like: /fen'bu:/)-X is a simple probalistic distribution library in JAX. The library is encouraged by *Distributions.jl*. In fenbux, We provide you:

* A simple and easy-to-use interface like **Distributions.jl**
* PyTree input/output
* Multiple dispatch for different distributions based on [plum-dispatch](https://github.com/beartype/plum)
* All jax feautures (vmap, pmap, jit, autograd etc.)

See [document](https://jiayaobo.github.io/fenbux/)

## Examples

### Statistics of Distributions 🤔

```python
import jax.numpy as jnp
from fenbux import variance, skewness, mean
from fenbux.univariate import Normal

μ = {'a': jnp.array([1., 2., 3.]), 'b': jnp.array([4., 5., 6.])} 
σ = {'a': jnp.array([4., 5., 6.]), 'b': jnp.array([7., 8., 9.])}

dist = Normal(μ, σ)
mean(dist) # {'a': Array([1., 2., 3.], dtype=float32), 'b': Array([4., 5., 6.], dtype=float32)}
variance(dist) # {'a': Array([16., 25., 36.], dtype=float32), 'b': Array([49., 64., 81.], dtype=float32)}
skewness(dist) # {'a': Array([0., 0., 0.], dtype=float32), 'b': Array([0., 0., 0.], dtype=float32)}
```

### Random Variables Generation

```python
import jax.random as jr
from fenbux import rand
from fenbux.univariate import Normal


key =  jr.PRNGKey(0)
x = {'a': {'c': {'d': {'e': 1.}}}}
y = {'a': {'c': {'d': {'e': 1.}}}}

dist = Normal(x, y)
rand(dist, key, shape=(3, )) # {'a': {'c': {'d': {'e': Array([1.6248107 , 0.69599575, 0.10169095], dtype=float32)}}}}
```

### Evaluations of Distribution 👩‍🎓

CDF, PDF, and more...

```python
import jax.numpy as jnp
from fenbux import cdf, logpdf
from fenbux.univariate import Normal


μ = jnp.array([1., 2., 3.])
σ = jnp.array([4., 5., 6.])

dist = Normal(μ, σ)
cdf(dist, jnp.array([1., 2., 3.])) # Array([0.5, 0.5, 0.5], dtype=float32)
logpdf(dist, jnp.array([1., 2., 3.])) # Array([-2.305233 , -2.5283763, -2.7106981], dtype=float32)
```

### Nested Transformations of Distribution 🤖

```python
import fenbux as fbx
import jax.numpy as jnp
from fenbux.univariate import Normal

# truncate and censor and affine
d = Normal(0, 1)
fbx.affine(fbx.censor(fbx.truncate(d, 0, 1), 0, 1), 0, 1)
fbx.logpdf(d, 0.5)
```

```
Array(-1.0439385, dtype=float32)
```

### Compatible with JAX transformations 😃

- vmap

```python
import jax.numpy as jnp
from jax import vmap

from fenbux import logpdf
from fenbux.univariate import Normal


dist = Normal({'a': jnp.zeros((2, 3))}, {'a':jnp.ones((2, 3, 5))}) # each batch shape is (2, 3)
x = jnp.zeros((2, 3, 5))
# claim use_batch=True to use vmap
vmap(logpdf, in_axes=(Normal(None, {'a': 2}, use_batch=True), 2))(dist, x) 
```

- grad

```python
import jax.numpy as jnp
from jax import jit, grad
from fenbux import logpdf
from fenbux.univariate import Normal

dist = Normal(0., 1.)
grad(logpdf)(dist, 0.)
```

### Speed 🔦
  
```python
import numpy as np
from scipy.stats import norm
from jax import jit
from fenbux import logpdf, rand
from fenbux.univariate import Normal
from tensorflow_probability.substrates.jax.distributions import Normal as Normal2

dist = Normal(0, 1)
dist2 = Normal2(0, 1)
dist3 = norm(0, 1)
x = np.random.normal(size=100000)

%timeit jit(logpdf)(dist, x).block_until_ready()
%timeit jit(dist2.log_prob)(x).block_until_ready()
%timeit dist3.logpdf(x)
```

```
51.2 µs ± 1.47 µs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
11.1 ms ± 176 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
1.12 ms ± 20.1 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
```

## Installation

* Install on your local device.

```bash
git clone https://github.com/JiaYaobo/fenbux.git
pip install -e .
```

* Install from PyPI.

```bash
pip install fenbux
```

## Reference

* [Distributions.jl](https://github.com/JuliaStats/Distributions.jl)
* [Equinox](https://github.com/patrick-kidger/equinox)
