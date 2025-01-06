import jax
from jaxtyping import Float

import fem.utils


@fem.utils.jit()
def right_cauchy_strain(F: Float[jax.Array, "3 3"]) -> Float[jax.Array, "3 3"]:
    r"""Cauchy strain tensor (right Cauchy-Green deformation tensor).

    $$
    \bm{C} = \bm{F}^T \bm{F}
    $$

    Args:
        F: Deformation gradient.

    Returns:
        Cauchy strain tensor (right Cauchy-Green deformation tensor).

    References:
        [1] <!-- --> [Finite strain theory - Wikipedia](https://en.wikipedia.org/wiki/Finite_strain_theory#Cauchy_strain_tensor_(right_Cauchy%E2%80%93Green_deformation_tensor))
    """
    return F.T @ F
