# Scipy With PyTree

In fenbux, we provide wrappers of `scipy.stats` for disributions, which accept pytrees as inputs and output pytrees.

```python
from fenbux import scipy_stats as stats

# Create a normal distribution
loc = {'a': 0.0, 'b': 1.0}
scale = {'a': 1.0, 'b': 2.0}
x = [0.0, 1.0]
normal = stats.norm(loc=loc, scale=scale)
normal.mean(), normal.pdf(x)
```

```
({'a': 0.0, 'b': 1.0},
 {'a': [0.3989422804014327, 0.24197072451914337],
  'b': [0.17603266338214976, 0.19947114020071635]})
```

