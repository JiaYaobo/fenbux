import os

import jax


os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

jax.config.update("jax_enable_x64", True)
