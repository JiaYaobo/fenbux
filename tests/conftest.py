import gc
import sys

import jax
import jax.config
import psutil
import pytest


jax.config.update("jax_enable_x64", True)


# Hugely hacky way of reducing memory usage in tests.
# JAX can be a little over-happy with its caching; this is especially noticable when
# performing tests and therefore doing an unusual amount of compilation etc.
# This can be enough to exceed the 8GB RAM available to Ubuntu instances on GitHub
# Actions.
@pytest.fixture(autouse=True)
def clear_caches():
    process = psutil.Process()
    if process.memory_info().vms > 4 * 2**30:  # >4GB memory usage
        jax.clear_backends()
        for module_name, module in sys.modules.copy().items():
            if module_name.startswith("jax"):
                if module_name not in ["jax.interpreters.partial_eval"]:
                    for obj_name in dir(module):
                        obj = getattr(module, obj_name)
                        if hasattr(obj, "cache_clear"):
                            try:
                                print(f"Clearing {obj}")
                                obj.cache_clear()
                            except Exception:
                                pass
        gc.collect()