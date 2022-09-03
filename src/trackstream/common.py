##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from types import MappingProxyType
from typing import Any, ItemsView, Iterator, KeysView, Mapping, TypeVar, ValuesView

__all__ = ["CollectionBase"]


##############################################################################
# TYPING

V = TypeVar("V")


##############################################################################
# CODE
##############################################################################


class CollectionBase(Mapping[str, V]):
    """Base class for a homogenous, keyed collection of objects.

    Parameters
    ----------
    data : dict[str, V] or None, optional
        Mapping of the data for the collection.
        If `None` (default) the collection is empty.
    name : str or None, optional keyword-only
        The name of the collection
    **kwargs : V, optional
        Further entries for the collection.
    """

    __slots__ = ("_data", "name")

    def __init__(self, data: Mapping[str, V] | None = None, /, *, name: str | None = None, **kwargs: V) -> None:
        d = dict(data) if data is not None else {}
        d.update(kwargs)

        self._data: dict[str, V]
        object.__setattr__(self, "_data", d)

        self.name: str | None
        object.__setattr__(self, "name", name)

    @property
    def data(self) -> MappingProxyType[str, V]:
        return MappingProxyType(self._data)

    def __getitem__(self, key: str) -> V:
        """Get 'key' from the data."""
        return self._data[key]

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._data!r})"

    def keys(self) -> KeysView[str]:
        return self._data.keys()

    def values(self) -> ValuesView[V]:
        return self._data.values()

    def items(self) -> ItemsView[str, V]:
        return self._data.items()

    @property
    def _k0(self) -> str:
        return next(iter(self._data.keys()))

    @property
    def _v0(self) -> V:
        return next(iter(self._data.values()))

    def __getattr__(self, key: str) -> MappingProxyType[str, Any]:
        """Map any unkown methods to the contained fields."""
        if key in ("__isabstractmethod__",):
            return object.__getattribute__(self, key)
        elif hasattr(self._v0, key):
            return MappingProxyType({k: getattr(v, key) for k, v in self.items()})
        raise AttributeError(f"{self.__class__.__name__!r} object has not attribute {key!r}")
