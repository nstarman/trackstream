from __future__ import annotations

# STDLIB
import copy
from functools import cached_property
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence, TypeVar, cast

# THIRD PARTY
from attrs import Attribute, Factory, fields

R = TypeVar("R")
T = TypeVar("T")


def convert_if_none(factory: Callable[[], R], *, deepcopy: bool = False) -> Callable[[R | None], R]:
    """Return default factory for optional input.

    Parameters
    ----------
    factory : callable[[], R]
        A callable taking no input arguments returning one argument of type
        ``R``.
    deepcopy : bool, optional
        Whether to deepcopy the input, by default `False`.

    Returns
    -------
    callable[[Optional[R]], R]
        A default factory.
    """

    def converter(value: R | None) -> R:
        """Check ``value``, calling ``factory`` if `None`.

        Parameters
        ----------
        value : ``R`` or None
            Value to check. If `None`, call ``factory``.

        Returns
        -------
        ``R``
        """
        out = factory() if value is None else value
        out = copy.deepcopy(out) if deepcopy else out
        return out

    return converter


def _drop_properties(cls: type, fields: Sequence[Attribute]) -> list[Attribute]:
    return [f for f in fields if not isinstance(getattr(cls, f.name, None), (property, cached_property))]


def _drop_fields_from(kls: type) -> Callable[[type, Sequence[Attribute]], list[Attribute]]:
    names = [f.name for f in fields(kls)]

    def drop(cls: type, fields: Sequence[Attribute]) -> list[Attribute]:
        return [f for f in fields if f.name in names]

    return drop


def _cache_factory(td: type) -> Callable[..., dict[str, Any]]:
    def factory(*_: Any) -> dict[str, Any]:
        return dict.fromkeys(td.__annotations__.keys())

    return factory


class FactoryTakesSelfProtocol(Protocol[T, R]):
    factory: Callable[[T], R]


def attrs_factory_decorator(func: Callable[[T], R]) -> FactoryTakesSelfProtocol[T, R]:
    return cast(FactoryTakesSelfProtocol, Factory(func, takes_self=True))


class _HasCache(Protocol):
    @property
    def _cache(self) -> Mapping:
        ...


@attrs_factory_decorator
def _cache_proxy_factory(self: _HasCache) -> MappingProxyType:
    return MappingProxyType(self._cache)
