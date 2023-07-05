# Fenbu-X: A Simple Statistical Distribution Library Based On PyTree in JAX

Fenbu-X is a simple statistical distribution library based on PyTree in JAX. You can use pytrees as parameters of distributions, and use plum-dispatch to dispatch methods of distributions.

* Exact Attributes of Distributions

```python
import jax.numpy as jnp
from fenbux import Normal, variance, skewness

x = {'a': {'c': {'d': {'e': 1.}}}}
y = {'a': {'c': {'d': {'e': 1.}}}}

dist = Normal(x, y)
variance(dist) # {'a': {'c': {'d': {'e': Array(4., dtype=float32)}}}}
skewness(dist) # {'a': {'c': {'d': {'e': Array(0., dtype=float32)}}}}
``` 

* Random Variables Generation

```python
import jax.random as jr
from fenbux import Normal, rand

key =  jr.PRNGKey(0)
x = {'a': {'c': {'d': {'e': 1.}}}}
y = {'a': {'c': {'d': {'e': 1.}}}}

dist = Normal(x, y)
rand(dist, key) # {'a': {'c': {'d': {'e': Array(-0.40088415, dtype=float32)}}}}
```

* Functions of Distribution

```python
import jax.numpy as jnp
from fenbux import Normal, cdf

x = {'a': {'c': {'d': {'e': 0.}}}}
y = {'a': {'c': {'d': {'e': 1.}}}}

dist = Normal(x, y)
cdf(dist, 0.) # {'a': {'c': {'d': {'e': Array(0.5, dtype=float32)}}}}
```

Installation

```bash
git clone https://github.com/JiaYaobo/fenbux.git
pip install -e .
```