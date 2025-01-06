import jax
import jax.numpy as jnp
from jaxtyping import Float

import fem.utils


@fem.utils.jit()
def deformation_gradient(
    disp: Float[jax.Array, "4 3"], grad_op: Float[jax.Array, "3 4"]
) -> Float[jax.Array, "3 3"]:
    """Compute the deformation gradient tensor for a tetrahedral element.

    References:
        - https://en.wikipedia.org/wiki/Finite_strain_theory#Deformation_gradient_tensor
    """
    grad: Float[jax.Array, "3 3"] = grad_op @ disp
    F: Float[jax.Array, "3 3"] = jnp.eye(3) + grad
    return F
