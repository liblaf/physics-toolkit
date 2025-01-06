import jax
import jax.numpy as jnp
from jaxtyping import Float

import fem.utils


@fem.utils.jit()
def grad_op(points: Float[jax.Array, "4 3"]) -> Float[jax.Array, "3 4"]:
    grad: Float[jax.Array, "3 4"] = jnp.linalg.pinv(
        jnp.vstack(
            [
                points[0] - points[3],
                points[1] - points[3],
                points[2] - points[3],
            ]
        )
    ) @ jnp.asarray(
        [
            [1, 0, 0, -1],
            [0, 1, 0, -1],
            [0, 0, 1, -1],
        ]
    )
    return grad
