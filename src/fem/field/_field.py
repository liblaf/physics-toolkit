import abc

import jax
import jax.numpy as jnp
from jaxtyping import Float

import fem


class Field(abc.ABC):
    region: fem.Region
    dim: int

    def __init__(self, region: fem.Region, dim: int) -> None:
        self.region = region
        self.dim = dim

    @property
    def n_dof(self) -> int:
        raise NotImplementedError

    def grad(self, values: Float[jax.Array, "c a i"]) -> Float[jax.Array, "i j q c"]:
        return jnp.einsum(
            "ca...,aJqc->...Jqc", values[self.region.mesh.cells], self.region.dhdX
        )

    def hess(self, values: Float[jax.Array, "c a i"]) -> Float[jax.Array, "i j k q c"]:
        raise NotImplementedError

    def interpolate(
        self, values: Float[jax.Array, "c a i"]
    ) -> Float[jax.Array, "i q c"]:
        return jnp.einsum(
            "ca...,aqc->...qc", values[self.region.mesh.cells], self.region.h
        )
