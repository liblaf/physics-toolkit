import jax
import pytest


@pytest.fixture(autouse=True)
def jax_config() -> None:
    jax.config.update("jax_debug_infs", True)
    jax.config.update("jax_debug_nans", True)
    jax.config.update("jax_enable_checks", True)
    jax.config.update("jax_enable_x64", True)
