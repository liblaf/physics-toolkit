from collections.abc import Callable, Iterable, Sequence
from typing import TypeVar, overload

_C = TypeVar("_C", bound=Callable)

@overload
def jit(
    fun: _C,
    static_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Iterable[str] | None = None,
    **kwargs,
) -> _C: ...
@overload
def jit(
    fun: None = None,
    static_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Iterable[str] | None = None,
    **kwargs,
) -> Callable[[_C], _C]: ...
