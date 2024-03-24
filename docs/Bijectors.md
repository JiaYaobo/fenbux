`fenbux` supports `tensorflow_probability`-like and `Bijectors.jl`-like bijector module.

For example, we can create a bijector and transform a sample as follows:

```python
import jax.numpy as jnp
from fenbux.bijector import Exp, evaluate

bij = Exp()
x = jnp.array([1., 2., 3.])
evaluate(bij, x)
```

And inverse transform a sample as follows:

```python
import jax.numpy as jnp
from fenbux.bijector import Exp, inverse, evaluate

bij = Exp()
y = jnp.array([1., 2., 3.])
evaluate(inverse(bij), y)
```

And you can apply a bijector to a distribution as follows:

```python
import jax.numpy as jnp
from fenbux.bijector import Exp, transform
from fenbux.univariate import Normal
from fenbux import logpdf

dist = Normal(0, 1)
bij = Exp()

log_normal = transform(dist, bij)

x = jnp.array([1., 2., 3.])
logpdf(log_normal, x)
```

Supported bijectors are:

::: fenbux.bijector.Exp

::: fenbux.bijector.Log

::: fenbux.bijector.Shift

::: fenbux.bijector.Scale

::: fenbux.bijector.Identity

::: fenbux.bijector.Logit

::: fenbux.bijector.LeakyReLU

::: fenbux.bijector.Reshape

::: fenbux.bijector.Inverse

::: fenbux.bijector.Chain
