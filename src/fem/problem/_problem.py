import abc
import functools
from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy.typing as npt
import scipy.optimize
from jaxtyping import Float


class Problem(abc.ABC):
    def fun(self, u: Float[npt.ArrayLike, " DoF"]) -> Float[jax.Array, ""]:
        u: Float[jax.Array, " DoF"] = jnp.asarray(u)
        f = self._fun(u)
        # ic(f)
        return f

    def jac(self, u: Float[npt.ArrayLike, " DoF"]) -> Float[jax.Array, " DoF"]:
        u: Float[jax.Array, " DoF"] = jnp.asarray(u)
        return self._jac(u)

    def hessp(
        self, u: Float[npt.ArrayLike, " DoF"], v: Float[npt.ArrayLike, " DoF"]
    ) -> Float[jax.Array, " DoF"]:
        u: Float[jax.Array, " DoF"] = jnp.asarray(u)
        v: Float[jax.Array, " DoF"] = jnp.asarray(v, dtype=u.dtype)
        return self._hessp(u, v)

    def solve(
        self,
        x0: Float[npt.ArrayLike, " DoF"] | None = None,
        method: str = "Newton-CG",
        options: Mapping[str, Any] = {"disp": True},
    ) -> scipy.optimize.OptimizeResult:
        x0: Float[jax.Array, " DoF"] = (
            jnp.zeros(self.n_dof) if x0 is None else jnp.asarray(x0)
        )
        result: scipy.optimize.OptimizeResult = scipy.optimize.minimize(
            self.fun,
            x0,
            method=method,
            jac=self.jac,
            hessp=self.hessp,
            options=options,
        )
        return result

    @abc.abstractmethod
    def _objective(self, u: Float[jax.Array, " DoF"]) -> Float[jax.Array, ""]: ...

    @functools.cached_property
    def n_dof(self) -> int:
        return self._n_dof()

    @abc.abstractmethod
    def _n_dof(self) -> int: ...

    @functools.cached_property
    def _fun(self) -> Callable[[Float[jax.Array, " DoF"]], Float[jax.Array, " DoF"]]:
        return jax.jit(self._objective)

    @functools.cached_property
    def _jac(self) -> Callable[[Float[jax.Array, " DoF"]], Float[jax.Array, " DoF"]]:
        return jax.jit(jax.grad(self._fun))

    @functools.cached_property
    def _hessp(
        self,
    ) -> Callable[
        [Float[jax.Array, " DoF"], Float[jax.Array, " DoF"]], Float[jax.Array, " DoF"]
    ]:
        def hessp(
            u: Float[jax.Array, " DoF"], v: Float[jax.Array, " DoF"]
        ) -> Float[jax.Array, " DoF"]:
            return jax.jvp(jax.grad(self._fun), (u,), (v,))[1]

        return jax.jit(hessp)
