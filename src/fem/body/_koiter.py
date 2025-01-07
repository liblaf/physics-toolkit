import felupe
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jaxtyping import Bool, Float

import fem


class BodyKoiter(fem.Body):
    free_mask: Bool[np.ndarray, " D"]
    mesh: felupe.Mesh
    values: Float[jax.Array, " D"]

    alpha: Float[jax.Array, " C"]
    beta: Float[jax.Array, " C"]
    pre_strain: Float[jax.Array, " C"]
    thickness: Float[jax.Array, " C"]

    def __init__(
        self,
        mesh: felupe.Mesh,
        free_mask: Bool[npt.ArrayLike, "#D"] = True,  # noqa: FBT002
        values: Float[npt.ArrayLike, "#D"] = 0.0,
        alpha: Float[npt.ArrayLike, "#C"] = 1e6,
        beta: Float[npt.ArrayLike, "#C"] = 1e4,
        pre_strain: Float[npt.ArrayLike, "#C"] = 1.0,
        thickness: Float[npt.ArrayLike, "#C"] = 1e-3,
    ) -> None:
        self.mesh = mesh
        self.free_mask = np.broadcast_to(np.asarray(free_mask), (mesh.ndof,))
        self.values = jnp.broadcast_to(jnp.asarray(values), (mesh.ndof,))
        self.alpha = jnp.broadcast_to(jnp.asarray(alpha), (self.n_cells,))
        self.beta = jnp.broadcast_to(jnp.asarray(beta), (self.n_cells,))
        self.pre_strain = jnp.broadcast_to(jnp.asarray(pre_strain), (self.n_cells,))
        self.thickness = jnp.broadcast_to(jnp.asarray(thickness), (self.n_cells,))

    @property
    def n_points(self) -> int:
        return self.mesh.npoints

    @property
    def n_cells(self) -> int:
        return self.mesh.ncells

    def _energy(self, u: Float[jax.Array, " D"]) -> Float[jax.Array, ""]:
        u: Float[jax.Array, "P 3"] = u.reshape((self.n_points, 3))
        u: Float[jax.Array, "C 3 3"] = u[self.mesh.cells]
        x: Float[jax.Array, "C 3 3"] = jnp.asarray(self.mesh.points[self.mesh.cells])
        W: Float[jax.Array, " C"] = jax.vmap(koiter)(
            u, x, self.alpha, self.beta, self.pre_strain, self.thickness
        )
        return jnp.sum(W)


def koiter(
    u: Float[jax.Array, "3 3"],
    x: Float[jax.Array, "3 3"],
    alpha: Float[jax.Array, ""],
    beta: Float[jax.Array, ""],
    pre_strain: Float[jax.Array, ""],
    thickness: Float[jax.Array, ""],
) -> Float[jax.Array, ""]:
    α: Float[jax.Array, ""] = alpha
    β: Float[jax.Array, ""] = beta
    h: Float[jax.Array, ""] = thickness
    I: Float[jax.Array, "2 2"] = fem.element.triangle.first_fundamental_form(x + u)
    Iu: Float[jax.Array, "2 2"] = (
        pre_strain * fem.element.triangle.first_fundamental_form(x)
    )
    M: Float[jax.Array, "2 2"] = jnp.linalg.pinv(Iu) @ I - jnp.eye(2)
    Ws: Float[jax.Array, "2 2"] = 0.5 * α * jnp.trace(M) ** 2 + β * jnp.trace(M @ M)
    W: Float[jax.Array, ""] = 0.5 * (0.025 * h * Ws * jnp.sqrt(jnp.linalg.det(Iu)))
    # W += 1.0 * jnp.mean(u, axis=0) * -jnp.asarray([0.0, -9.8, 0.0])
    return W
