"""Cache descriptors and Protocols."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar, overload

__all__: list[str] = []

if TYPE_CHECKING:
    from collections.abc import Mapping


class _HasInitCache(Protocol):
    def _init_cache(self: Any, instance: Any) -> dict[str, Any]:
        ...


Self = TypeVar("Self", bound=_HasInitCache)  # from typing_extensions import Self


class _SupportsCache(Protocol):
    _cache: dict[str, Any]


EnclT = TypeVar("EnclT", bound="_SupportsCache")


class CacheProperty(Generic[EnclT]):
    """Add an attribute ``cache`` to a class."""

    @overload
    def __get__(self: Any, instance: EnclT, _: None | type) -> MappingProxyType[str, Any]:
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

        return MappingProxyType(instance._cache)  # noqa: SLF001

    def __set__(self: Any, instance: EnclT, value: Mapping[str, Any] | None) -> None:
        if not hasattr(instance, "_cache"):
            cache = self._init_cache(instance)
            cache.update(value or {})
            object.__setattr__(instance, "_cache", cache)
        else:
            msg = "can't set attribute"
            raise AttributeError(msg)

    def __delete__(self: Any, instance: EnclT) -> None:
        instance._cache.clear()  # noqa: SLF001
        instance._cache.update(self._init_cache(instance))  # noqa: SLF001

    @staticmethod
    def _init_cache(instance: EnclT) -> dict[str, Any]:
        """Initiallize cache on enclosing instance."""
        cache_cls = getattr(instance, "_CACHE_CLS", {})
        return dict.fromkeys(getattr(cache_cls, "__annotations__", {}).keys())
