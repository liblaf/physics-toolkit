import abc
import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Bool, Float


class Body(abc.ABC):
    free_mask: Bool[np.ndarray, "*D"]
    values: Float[jax.Array, "*D"]

    def make_field(self, u: Float[jax.Array, " DoF"]) -> Float[jax.Array, "*D"]:
        u: Float[jax.Array, "*D"] = self.values.at[self.free_mask].set(u)
        return u

    @functools.cached_property
    def n_degrees(self) -> int:
        return self.free_mask.size

    @functools.cached_property
    def n_dof(self) -> int:
        return jnp.count_nonzero(self.free_mask).item()

    @functools.cached_property
    def fun(self) -> Callable[[Float[jax.Array, " DoF"]], Float[jax.Array, ""]]:
        return jax.jit(self._fun)

    @functools.cached_property
    def jac(self) -> Callable[[Float[jax.Array, " DoF"]], Float[jax.Array, " DoF"]]:
        return jax.jit(jax.grad(self._fun))

    @functools.cached_property
    def hessp(
        self,
    ) -> Callable[
        [Float[jax.Array, " DoF"], Float[jax.Array, " DoF"]],
        Float[jax.Array, " DoF"],
    ]:
        def hessp(
            u: Float[jax.Array, " DoF"], v: Float[jax.Array, " DoF"]
        ) -> Float[jax.Array, " DoF"]:
            return jax.jvp(jax.grad(self._fun), (u,), (v,))[1]

        return jax.jit(hessp)

    def _fun(self, u: Float[jax.Array, " DoF"]) -> Float[jax.Array, ""]:
        u: Float[jax.Array, "*D"] = self.make_field(u)
        return self._energy(u)

    @abc.abstractmethod
    def _energy(self, u: Float[jax.Array, "*D"]) -> Float[jax.Array, ""]: ...
