import jax
import jax.numpy as jnp
from jaxtyping import Float

import fem.utils


@fem.utils.jit()
def volume(points: Float[jax.Array, "4 3"]) -> Float[jax.Array, ""]:
    """Compute the volume of a tetrahedral element.

    Args:
        points: The coordinates of the four vertices of the tetrahedron.

    Returns:
        The volume of the tetrahedron.

    References:
        - https://en.wikipedia.org/wiki/Tetrahedron#Volume
    """
    volume: Float[jax.Array, ""] = jnp.linalg.det(points[1:] - points[0]) / 6.0
    volume = jnp.abs(volume)
    return volume
