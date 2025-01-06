import abc
import functools
from collections.abc import Callable

import jax
from jaxtyping import Bool, Float


class Body(abc.ABC):
    free_mask: Bool[jax.Array, " DoF"]
    values: Float[jax.Array, " DoF"]

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

    @abc.abstractmethod
    def _fun(self, u: Float[jax.Array, " DoF"]) -> Float[jax.Array, " "]: ...
