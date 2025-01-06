from collections.abc import Iterable, Mapping
from typing import Any

import jax
import jax.numpy as jnp


def as_dict_of_array(
    data: Mapping[str, Any], filter_keys: Iterable[str] | None = None
) -> dict[str, jax.Array]:
    if filter_keys is None:
        return {k: jnp.asarray(v) for k, v in data.items()}
    filter_keys = set(filter_keys)
    return {k: jnp.asarray(v) for k, v in data.items() if k in filter_keys}
