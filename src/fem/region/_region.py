import functools

import felupe
import jax
import jax.numpy as jnp
from jaxtyping import Float


class Region:
    _region: felupe.Region
    element: felupe.element.Element
    mesh: felupe.Mesh
    quadrature: felupe.quadrature.Scheme

    def __init__(
        self,
        mesh: felupe.Mesh,
        element: felupe.element.Element,
        quadrature: felupe.quadrature.Scheme,
    ) -> None:
        self._region = felupe.Region(mesh, element, quadrature)

    @property
    def mesh(self) -> felupe.Mesh:
        return self._region.mesh  # pyright: ignore [reportAttributeAccessIssue]

    @property
    def element(self) -> felupe.element.Element:
        return self._region.element  # pyright: ignore [reportAttributeAccessIssue]

    @property
    def quadrature(self) -> felupe.quadrature.Scheme:
        return self._region.quadrature  # pyright: ignore [reportAttributeAccessIssue]

    @functools.cached_property
    def h(self) -> Float[jax.Array, "a q 1"]:
        return jnp.asarray(self._region.h)  # pyright: ignore [reportAttributeAccessIssue]

    @functools.cached_property
    def dhdr(self) -> Float[jax.Array, "a J q 1"]:
        return jnp.asarray(self._region.dhdr)  # pyright: ignore [reportAttributeAccessIssue]

    @functools.cached_property
    def dXdr(self) -> Float[jax.Array, "I J q c"]:
        return jnp.asarray(self._region.dXdr)  # pyright: ignore [reportAttributeAccessIssue]

    @functools.cached_property
    def drdX(self) -> Float[jax.Array, "J I q c"]:
        return jnp.asarray(self._region.drdX)  # pyright: ignore [reportAttributeAccessIssue]

    @functools.cached_property
    def dV(self) -> Float[jax.Array, "q c"]:
        return jnp.asarray(self._region.dV)  # pyright: ignore [reportAttributeAccessIssue]

    @functools.cached_property
    def dhdX(self) -> Float[jax.Array, "a J q c"]:
        return jnp.asarray(self._region.dhdX)  # pyright: ignore [reportAttributeAccessIssue]
