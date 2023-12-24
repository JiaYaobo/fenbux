# Batch With VMap

In `JAX`, we can use `vmap` to vectorize a function. For example,

```python
import jax.numpy as jnp
from jax import vmap

def f(x):
    return x ** 2

vmap(f)(jnp.arange(3)) # Array([0, 1, 4], dtype=int32)
```

And we can specify the axis to vectorize on, namely a batch axis. For example,

```python
import jax.numpy as jnp
from jax import vmap

def f(x):
    return x ** 2

vmap(f, in_axes=(0,))(jnp.arange(3)) # Array([0, 1, 4], dtype=int32)
```

However, it's difficult to specify batch axis for a customized PyTree node. In `fenbux`, every distribution is treated as a PyTree, user can use `use_batch=True` to specify the batch axis . For example,

```python
import jax.numpy as jnp
from jax import vmap

from fenbux import Normal, logpdf

dist = Normal(0, jnp.ones((2, 3, 5))) # each batch shape is (2, 3)
x = jnp.zeros((2, 3, 5))
# set claim use_batch=True to use vmap
vmap(logpdf, in_axes=(Normal(None, 2, use_batch=True), 2))(dist, x) 
```

Here `Normal(None, 0, use_batch=True)` means that we don't care about the batch axis of the first argument `mean`, and we want to vectorize on the second argument `sd` on 3rd dimension, namely the batch dimension.

