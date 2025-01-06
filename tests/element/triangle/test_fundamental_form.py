import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Float, PRNGKeyArray

import fem.element.triangle


def test_first_fundamental_form() -> None:
    key: PRNGKeyArray = jax.random.key(0)
    points: Float[jax.Array, "3 3"] = jax.random.uniform(key, (3, 3))
    I: Float[jax.Array, "2 2"] = fem.element.triangle.first_fundamental_form(points)
    I_desired: Float[jax.Array, "2 2"] = first_fundamental_form_naive(points)
    np.testing.assert_allclose(I, I_desired)


def first_fundamental_form_naive(
    points: Float[jax.Array, "3 3"],
) -> Float[jax.Array, "2 2"]:
    """Chen, Zhen, et al. "Fine wrinkling on coarsely meshed thin shells." ACM Transactions on Graphics (TOG) 40.5 (2021): 1-32."""
    vi: Float[jax.Array, 3]
    vj: Float[jax.Array, 3]
    vk: Float[jax.Array, 3]
    vi, vj, vk = points
    I: Float[jax.Array, "2 2"] = jnp.asarray(
        [
            [jnp.sum((vj - vi) ** 2), jnp.dot(vj - vi, vk - vi)],
            [jnp.dot(vk - vi, vj - vi), jnp.sum((vk - vi) ** 2)],
        ]
    )
    return I
