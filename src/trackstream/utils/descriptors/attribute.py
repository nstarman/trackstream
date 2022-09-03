from __future__ import annotations

# STDLIB
from copy import deepcopy
from dataclasses import dataclass, field
from typing import (
    Any,
    Generic,
    Literal,
    MutableMapping,
    NoReturn,
    TypeVar,
    final,
    overload,
)

T = TypeVar("T")


@final
@dataclass(frozen=True)
class Attribute(Generic[T]):
    """An attribute on an instance, with optional caching.

    Parameters
    ----------
    obj : T
        The attribute object.
    attrs_loc : {'__dict__', '_attrs_', None}, optional
        Where the attribute is cached, by default on the instance's dict.
    """

    obj: T
    attrs_loc: Literal["__dict__", "_attrs_"] | None = field(default="__dict__")

    def __post_init__(self) -> None:
        object.__setattr__(self, "__doc__", getattr(self.obj, "__doc__", None))

    def __set_name__(self, _: type, name: str) -> None:
        self._enclosing_attr: str
        object.__setattr__(self, "_enclosing_attr", name)

    # TODO! use Self, not Attribute[T] when that works
    @overload
    def __get__(self: Attribute[T], instance: None, _: Any) -> Attribute[T]:
        ...

    @overload
    def __get__(self: Attribute[T], instance: object, _: Any) -> T:
        ...

    def __get__(self: Attribute[T], instance: object | None, _: Any) -> Attribute[T] | T:
        # Get from class:
        if instance is None:
            return self

        # Get from instance:
        obj: T
        if self.attrs_loc is None:
            obj = deepcopy(self.obj)
        else:
            if not hasattr(instance, self.attrs_loc):
                object.__setattr__(instance, self.attrs_loc, dict())

            attrs: MutableMapping[str, Any] = getattr(instance, self.attrs_loc)
            maybeobj = attrs.get(self._enclosing_attr)  # get from enclosing.

            if maybeobj is None:
                obj = attrs[self._enclosing_attr] = deepcopy(self.obj)
            else:
                obj = maybeobj

        return obj

    def __set__(self, _: str, __: Any) -> NoReturn:
        raise AttributeError("can't set attribute")
