##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import inspect
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

__all__ = ["NumPyOverloader", "TypeInfo"]


##############################################################################
# TYPING

Self = TypeVar("Self")


##############################################################################
# CODE
##############################################################################


@final
@dataclass(frozen=True)
class TypeInfo:
    """Info on the type of each argument."""

    type: type
    allow_subclass: bool = True

    def validate_type(self, arg_type: type) -> bool:
        """Validate the argument type."""
        isvalid: bool
        if self.allow_subclass:
            isvalid = issubclass(arg_type, self.type)
        else:
            isvalid = arg_type is self.type

        return isvalid


@final
@dataclass(frozen=True)
class NumPyInfo:
    """Info to overload a :mod:`numpy` function."""

    func: Callable[..., Any]
    """The overloading function."""

    implements: Callable[..., Any]
    """The overloaded :mod:`numpy` function."""

    types: TypeInfo | tuple[TypeInfo, ...]
    """
    The argument types for the overloaded function.
    Used to check if the overload is valid.
    """

    def validate_types(self, types: tuple[type, ...]) -> bool:
        """Check the types of the arguments.

        Parameters
        ----------
        types : tuple[type, ...]
            Tuple of the types of the arguments.

        Returns
        -------
        bool
            Whether the argument types work for the ``func``.
        """
        # Construct types to check.
        if isinstance(self.types, tuple):
            valid_types = self.types
        else:  # one type applies to all arguments
            valid_types = tuple(self.types for _ in types)

        # Check that each argument is the correct type.
        isvalid: bool = True
        for i, t in enumerate(types):
            vt = valid_types[i]
            isvalid = vt.validate_type(t)

            if not isvalid:
                break

        return isvalid

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

    def __call__(self, obj) -> NumPyInfo:
        """Run dispatcher."""
        return self._dispatcher(obj)


##############################################################################


@dataclass(frozen=True)
class NumPyOverloader(Mapping[Callable[..., Any], Dispatcher]):
    """Overload :mod:`numpy` functions.

    Parameters
    ----------
    default_dispatch_on : type or None, optional
        Default type to dispatch on.
    _reg : dict[callable[..., Any], Dispatcher]
        Registry of overloads.
    """

    default_dispatch_on: type | None = None

    _reg: dict[Callable[..., Any], Dispatcher] = field(default_factory=dict, repr=False)

    # ===============================================================
    # Mapping

    def __getitem__(self, key: Callable[..., Any]) -> Dispatcher:
        return self._reg[key]

    def __iter__(self) -> Iterator[Callable[..., Any]]:
        return iter(self._reg)

    def __len__(self) -> int:
        return len(self._reg)

    def keys(self) -> KeysView[Callable[..., Any]]:
        return self._reg.keys()

    def values(self) -> ValuesView[Dispatcher]:
        return self._reg.values()

    def __contains__(self, o: object) -> bool:
        return o in self._reg

    # ===============================================================

    def implements(
        self,
        numpy_function: Callable[..., Any],
        /,
        *,
        types: type | TypeInfo | tuple[type | TypeInfo, ...],
        dispatch_on: type | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register an __array_function__ implementation object."""
        # Get dispatch type
        dispatch_on = self.default_dispatch_on if dispatch_on is None else dispatch_on
        if dispatch_on is None:
            raise ValueError("no default dispatch type -- need to give one.")
        else:
            dispatch_type = dispatch_on

        # Turn ``types`` into only TypeInfo
        if isinstance(types, TypeInfo):
            tinfo = types
        elif inspect.isclass(types):
            tinfo = TypeInfo(types)
        elif isinstance(types, tuple):
            tinfo = tuple(t if isinstance(t, TypeInfo) else TypeInfo(t) for t in types)
        else:
            raise ValueError(f"types must be a {self.implements.__annotations__['types']}")

        # Make single-dispatcher for numpy function
        if numpy_function not in self._reg:
            self._reg[numpy_function] = Dispatcher()

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            """Add function to numpy overloading.

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
            info = NumPyInfo(func=func, types=tinfo, implements=numpy_function)
            # Register the function
            self._reg[numpy_function]._dispatcher.register(dispatch_type, info)
            return func

        return decorator
