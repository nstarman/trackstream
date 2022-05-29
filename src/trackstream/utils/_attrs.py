# -*- coding: utf-8 -*-


# STDLIB
import copy
from functools import cached_property
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    cast,
)

# THIRD PARTY
from attrs import Attribute, Factory, fields

T = TypeVar("T")
R = TypeVar("R")


def convert_if_none(
    factory: Callable[..., R], *, deepcopy: bool = False
) -> Callable[[Optional[R]], R]:
    def converter(value: Optional[R]) -> R:
        out = factory() if value is None else value
        out = copy.deepcopy(out) if deepcopy else out
        return out

    return converter


def _drop_properties(cls: type, fields: Sequence[Attribute]) -> List[Attribute]:
    return [
        f for f in fields if not isinstance(getattr(cls, f.name, None), (property, cached_property))
    ]


def _drop_fields_from(kls: type) -> Callable[[type, Sequence[Attribute]], List[Attribute]]:
    names = [f.name for f in fields(kls)]

    def drop(cls: type, fields: Sequence[Attribute]) -> List[Attribute]:
        return [f for f in fields if f.name in names]

    return drop


def _cache_factory(td: type) -> Callable[..., Dict[str, Any]]:
    def factory(*_: Any) -> Dict[str, Any]:
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
