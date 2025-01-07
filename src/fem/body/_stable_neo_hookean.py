import felupe
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jaxtyping import Bool, Float

import fem


class BodyStableNeoHookean(fem.Body):
    free_mask: Bool[np.ndarray, " D"]
    mesh: felupe.Mesh
    values: Float[jax.Array, " D"]

    lambda_: Float[jax.Array, " C"]
    mu: Float[jax.Array, " C"]
    pre_strain: Float[jax.Array, " C"]

    def __init__(
        self,
        mesh: felupe.Mesh,
        free_mask: Bool[npt.ArrayLike, "#D"] = True,  # noqa: FBT002
        values: Float[npt.ArrayLike, "#D"] = 0.0,
        lambda_: Float[npt.ArrayLike, "#C"] = 1e4,
        mu: Float[npt.ArrayLike, "#C"] = 1e2,
        pre_strain: Float[npt.ArrayLike, "#C"] = 1.0,
    ) -> None:
        self.mesh = mesh
        self.free_mask = np.broadcast_to(np.asarray(free_mask), (mesh.ndof,))
        self.values = jnp.broadcast_to(jnp.asarray(values), (mesh.ndof,))
        self.lambda_ = jnp.broadcast_to(jnp.asarray(lambda_), (self.n_cells,))
        self.mu = jnp.broadcast_to(jnp.asarray(mu), (self.n_cells,))
        self.pre_strain = jnp.broadcast_to(jnp.asarray(pre_strain), (self.n_cells,))

    @property
    def n_points(self) -> int:
        return self.mesh.npoints

    @property
    def n_cells(self) -> int:
        return self.mesh.ncells

    def _energy(self, u: Float[jax.Array, " D"]) -> Float[jax.Array, ""]:
        u: Float[jax.Array, "P 3"] = u.reshape((self.n_points, 3))
        u: Float[jax.Array, "C 4 3"] = u[self.mesh.cells]
        x: Float[jax.Array, "C 4 3"] = jnp.asarray(self.mesh.points[self.mesh.cells])
        W: Float[jax.Array, " C"] = jax.vmap(stable_neo_hookean)(
            u, x, self.lambda_, self.mu, self.pre_strain
        )
        return jnp.sum(W)


def stable_neo_hookean(
    u: Float[jax.Array, "4 3"],
    x: Float[jax.Array, "4 3"],
    lambda_: Float[jax.Array, ""],
    mu: Float[jax.Array, ""],
    pre_strain: Float[jax.Array, ""],
) -> Float[jax.Array, ""]:
    λ: Float[jax.Array, ""] = lambda_
    μ: Float[jax.Array, ""] = mu
    α: Float[jax.Array, ""] = 1.0 + μ / λ - (μ / 4.0) * λ
    grad_op: Float[jax.Array, "3 4"] = fem.element.tetra.grad_op(x)
    grad: Float[jax.Array, "3 3"] = grad_op @ u
    F: Float[jax.Array, "3 3"] = jnp.eye(3) / pre_strain + grad
    J: Float[jax.Array, ""] = jnp.linalg.det(F)  # relative volume change
    C: Float[jax.Array, "3 3"] = F.T @ F
    I_C: Float[jax.Array, "3 3"] = jnp.trace(C)  # first invariant of C
    # W: Float[jax.Array, ""] = (
    #     0.5 * μ * (I_C - 3) + 0.5 * λ * (J - α) ** 2 - 0.5 * μ * jnp.log(I_C + 1)
    # )
    # TODO: verify the following implementation
    λ_hat: Float[jax.Array, ""] = μ + λ
    W: Float[jax.Array, ""] = (
        0.5 * μ * (I_C - 3) + 0.5 * λ_hat * (J - 1 - μ / λ_hat) ** 2
    )
    W *= fem.element.tetra.volume(x)
    return W


def linear_elasticity(
    u: Float[jax.Array, "4 3"],
    x: Float[jax.Array, "4 3"],
    lambda_: Float[jax.Array, ""],
    mu: Float[jax.Array, ""],
) -> Float[jax.Array, ""]:
    λ: Float[jax.Array, ""] = lambda_
    μ: Float[jax.Array, ""] = mu
    grad_op: Float[jax.Array, "3 4"] = fem.element.tetra.grad_op(x)
    grad: Float[jax.Array, "3 3"] = grad_op @ u
    E: Float[jax.Array, "3 3"] = 0.5 * (grad.T @ grad)
    W: Float[jax.Array, ""] = 0.5 * λ * jnp.trace(E) ** 2 + μ * jnp.trace(E @ E)
    W *= fem.element.tetra.volume(x)
    return W
