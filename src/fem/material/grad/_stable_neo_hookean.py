from collections.abc import Mapping

import jax
import jax.numpy as jnp
from jaxtyping import Float

import fem


class StableNeoHookean(fem.material.MaterialBasedOnGradient):
    required_keys: frozenset[str] = frozenset({"lambda", "mu"})

    def _energy_density(
        self, grad: jax.Array, cell_data: Mapping[str, jax.Array]
    ) -> jax.Array:
        return stable_neo_hookean(grad, cell_data["lambda"], cell_data["mu"])


def stable_neo_hookean(
    grad: Float[jax.Array, "3 3"],
    lambda_: Float[jax.Array, ""],
    mu: Float[jax.Array, ""],
) -> Float[jax.Array, ""]:
    r"""Stable Neo-Hookean.

    Args:
        grad: Gradient of displacement field.
        lambda_: ($\lambda$) Lamé's first parameter.
        mu: ($\mu$) Shear modulus.

    Returns:
        Energy density.

    References:
        [1] Smith, Breannan, Fernando De Goes, and Theodore Kim. "Stable neo-hookean flesh simulation." ACM Transactions on Graphics (TOG) 37.2 (2018): 1-15.
    """
    F: Float[jax.Array, "3 3"] = jnp.eye(3) + grad
    λ: Float[jax.Array, ""] = lambda_
    μ: Float[jax.Array, ""] = mu
    J: Float[jax.Array, ""] = jnp.linalg.det(F)  # relative volume change
    C: Float[jax.Array, "3 3"] = F.T @ F
    I_C: Float[jax.Array, "3 3"] = jnp.trace(C)  # first invariant of C
    α: Float[jax.Array, ""] = 1 + μ / λ - (μ / 4) * λ
    W: Float[jax.Array, ""] = (
        0.5 * μ * (I_C - 3) + 0.5 * λ * (J - α) ** 2 - 0.5 * μ * jnp.log(I_C + 1)
    )
    return W
