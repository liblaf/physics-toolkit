import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Float, PRNGKeyArray

import fem.element.tetra


def test_deformation_gradient() -> None:
    rng: PRNGKeyArray = jax.random.key(0)
    points: Float[jax.Array, "4 3"] = jax.random.uniform(rng, (4, 3))
    disp: Float[jax.Array, "4 3"] = jax.random.uniform(rng, (4, 3))
    F: Float[jax.Array, "3 3"] = fem.element.tetra.deformation_gradient(
        disp, fem.element.tetra.grad_op(points)
    )
    np.testing.assert_allclose(F, deformation_gradient_naive(disp, points), atol=1e-15)


def deformation_gradient_naive(
    disp: Float[jax.Array, "4 3"], points: Float[jax.Array, "4 3"]
) -> Float[jax.Array, "3 3"]:
    """https://en.wikipedia.org/wiki/Finite_strain_theory."""
    x: Float[jax.Array, "4 3"] = points + disp
    dX: Float[jax.Array, "3 3"] = points[1:] - points[0]
    dx: Float[jax.Array, "3 3"] = x[1:] - x[0]
    F: Float[jax.Array, "3 3"] = dx @ jnp.linalg.pinv(dX)
    return F
