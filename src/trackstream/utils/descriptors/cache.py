"""Cache descriptors and Protocols."""

from __future__ import annotations

# STDLIB
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Generic, Protocol, TypeVar, overload


class _HasInitCache(Protocol):
    def _init_cache(self, instance: Any) -> dict[str, Any]:
        ...


Self = TypeVar("Self", bound=_HasInitCache)  # from typing_extensions import Self


class _SupportsCache(Protocol):
    _cache: dict[str, Any]


EnclT = TypeVar("EnclT", bound="_SupportsCache")


class CacheProperty(Generic[EnclT]):
    """Add an attribute ``cache`` to a class."""

    @overload
    def __get__(self, instance: EnclT, _: None | type) -> MappingProxyType[str, Any]:
        ...

    @overload
    def __get__(self: Self, instance: None, _: None | type) -> Self:
        ...

    def __get__(self: Self, instance: EnclT | None, _: None | type) -> MappingProxyType[str, Any] | Self:
        if instance is None:
            return self

        if not hasattr(instance, "_cache"):
            cache = self._init_cache(instance)
            object.__setattr__(instance, "_cache", cache)

        return MappingProxyType(instance._cache)

    def __set__(self, instance: EnclT, value: Mapping[str, Any] | None) -> None:
        if not hasattr(instance, "_cache"):
            cache = self._init_cache(instance)
            cache.update(value or {})
            object.__setattr__(instance, "_cache", cache)
        else:
            raise AttributeError("can't set attribute")

    def __delete__(self, instance: EnclT) -> None:
        instance._cache.clear()
        instance._cache.update(self._init_cache(instance))

    @staticmethod
    def _init_cache(instance: EnclT) -> dict[str, Any]:
        """Initiallize cache on enclosing instance."""
        cache_cls = getattr(instance, "_CACHE_CLS", {})
        cache = dict.fromkeys(getattr(cache_cls, "__annotations__", {}).keys())
        return cache
