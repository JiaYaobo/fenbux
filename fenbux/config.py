from jax import config


def use_x64():
    """Use 64-bit floating point precision"""
    config.update("jax_enable_x64", True)
