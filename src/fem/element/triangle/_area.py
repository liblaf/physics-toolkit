import jax
import jax.numpy as jnp
from jaxtyping import Float

import fem.utils


@fem.utils.jit()
def area(points: Float[jax.Array, "3 3"]) -> Float[jax.Array, ""]:
    return 0.5 * jnp.linalg.norm(
        jnp.cross(points[0] - points[2], points[1] - points[2])
    )
