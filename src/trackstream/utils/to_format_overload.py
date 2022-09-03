##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from dataclasses import dataclass, field
from functools import singledispatch
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Iterator,
    KeysView,
    Mapping,
    NoReturn,
    TypeVar,
    ValuesView,
    final,
)

__all__ = ["ToFormatOverloader"]


##############################################################################
# CODE

Self = TypeVar("Self")


##############################################################################
# CODE
##############################################################################


@final
@dataclass(frozen=True)
class ToFormatInfo:
    """Info to overload a :mod:`numpy` function."""

    func: Callable[..., Any]
    """The to_format function."""

    implements: type
    """The type ``to_format`` implements."""

    def __call__(self: Self, *args: Any) -> Self:
        """Return self. Used for singledispatch in Dispatcher."""
        return self


@final
class Dispatcher:
    """Single dispatch."""

    def __init__(self) -> None:
        @singledispatch
        def dispatcher(obj: object) -> NoReturn:
            raise NotImplementedError("not dispatched")

        self._dispatcher = dispatcher

    @property
    def registry(self) -> MappingProxyType[Any, Callable[..., Any]]:
        return self._dispatcher.registry

    def __call__(self, obj) -> ToFormatInfo:
        """Run dispatcher."""
        return self._dispatcher(obj)


##############################################################################


@final
@dataclass(frozen=True)
class ToFormatOverloader(Mapping[type, Dispatcher]):

    default_dispatch_on: type | None = None

    _reg: dict[type, Dispatcher] = field(default_factory=dict, repr=False)

    # ===============================================================
    # Mapping

    def __getitem__(self, key: type) -> Dispatcher:
        return self._reg[key]

    def __iter__(self) -> Iterator[type]:
        return iter(self._reg)

    def __len__(self) -> int:
        return len(self._reg)

    def keys(self) -> KeysView[type]:
        return self._reg.keys()

    def values(self) -> ValuesView[Dispatcher]:
        return self._reg.values()

    def __contains__(self, o: object) -> bool:
        return o in self._reg

    # ===============================================================

    def implements(
        self,
        type: type,
        /,
        *,
        dispatch_on: type | None = None,
    ):
        """Register a ``to_format`` implementation."""
        # Get dispatch type
        dispatch_on = self.default_dispatch_on if dispatch_on is None else dispatch_on
        if dispatch_on is None:
            raise ValueError("no default dispatch type -- need to give one.")
        else:
            dispatch_type = dispatch_on

        # Make single-dispatcher for numpy function
        if type not in self._reg:
            self._reg[type] = Dispatcher()

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            """Add function to ``to_format`` overloading.

            Parameters
            ----------
            func : Callable[..., Any]
                The function.

            Returns
            -------
            func : Callable[..., Any]
                Same as ``func``.
            """
            # Adding a new numpy function
            info = ToFormatInfo(func=func, implements=type)
            # Register the function
            self._reg[type]._dispatcher.register(dispatch_type, info)
            return func

        return decorator
