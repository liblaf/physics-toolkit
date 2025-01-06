import abc
import functools
from collections.abc import Callable, Mapping
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import numpy.typing as npt
from jaxtyping import Float

import fem


class EnergyDensityFn(Protocol):
    def __call__(
        self, grad: jax.Array, properties: Mapping[str, jax.Array]
    ) -> Float[jax.Array, ""]: ...


class EnergyDensityFnLike(Protocol):
    def __call__(
        self, grad: npt.ArrayLike, properties: Mapping[str, Any]
    ) -> Float[jax.Array, ""]: ...


class MaterialBasedOnGradient(abc.ABC):
    required_keys: frozenset[str]

    @abc.abstractmethod
    def _energy_density(
        self, grad: Float[jax.Array, "3 3"], properties: Mapping[str, jax.Array]
    ) -> Float[jax.Array, ""]: ...

    def __call__(
        self, grad: npt.ArrayLike, properties: Mapping[str, Any]
    ) -> Float[jax.Array, ""]:
        return self.fun(grad, properties)

    @functools.cached_property
    def fun(self) -> EnergyDensityFnLike:
        return self.wraps()(self._energy_density)

    @functools.cached_property
    def vmap(self) -> EnergyDensityFnLike:
        return self.wraps(vmap=True)(self._energy_density)

    def wraps(
        self, *, jit: bool = True, vmap: bool = False
    ) -> Callable[[EnergyDensityFn], EnergyDensityFnLike]:
        def wrapper(fn: EnergyDensityFn) -> EnergyDensityFnLike:
            if vmap:
                fn = jax.vmap(fn)
            if jit:
                fn = jax.jit(fn)

            @functools.wraps(fn)
            def wrapped(
                grad: npt.ArrayLike, properties: Mapping[str, Any]
            ) -> Float[jax.Array, ""]:
                grad: jax.Array = jnp.asarray(grad)
                properties: dict[str, jax.Array] = fem.utils.as_dict_of_array(
                    properties, self.required_keys
                )
                return fn(grad, properties)

            return wrapped

        return wrapper
