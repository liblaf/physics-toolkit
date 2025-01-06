import functools
from collections.abc import Callable, Iterable, Sequence
from typing import TypeVar

import jax

_C = TypeVar("_C", bound=Callable)


def jit(
    fun: _C | None = None,
    static_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Iterable[str] | None = None,
    **kwargs,
) -> _C | Callable[[_C], _C]:
    if fun is None:
        return functools.partial(
            jax.jit,
            static_argnums=static_argnums,
            static_argnames=static_argnames,
            **kwargs,
        )  # pyright: ignore [reportReturnType]
    return jax.jit(
        fun,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
        **kwargs,
    )
