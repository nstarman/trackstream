"""Common code for trackstream."""

from __future__ import annotations

# STDLIB
from collections.abc import ItemsView, Iterator, KeysView, Mapping, ValuesView
from types import MappingProxyType
from typing import Any, TypeVar

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
        Mapping of the data for the collection. If `None` (default) the
        collection is empty.
    name : str or None, optional keyword-only
        The name of the collection
    **kwargs : V, optional
        Further entries for the collection.

    Examples
    --------
    Let's do a pretty trivial example: a collection of `~numpy.ndarray`.

        >>> a = np.arange(5)
        >>> b = np.arange(5, 10)

    Collecting it in a `~trackstream.common.CollectionBase`.

        >>> from trackstream.common import CollectionBase
        >>> collection = CollectionBase({"a": a, "b": b})

    A collection is a `collections.abc.Mapping` (immutable) to the data.

        >>> collection.data
        mappingproxy({'a': array([0, 1, 2, 3, 4]), 'b': array([5, 6, 7, 8, 9])})
        >>> collection["a"]
        array([0, 1, 2, 3, 4])

    Collections can have a name.

        >>> ex = CollectionBase({}, name='Example')
        >>> ex.name
        'Example'

    The contained type is homogeneous, allowing
    `~trackstream.common.CollectionBase` to reach into each field.

        >>> collection.dtype
        mappingproxy({'a': dtype('int64'), 'b': dtype('int64')})
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
        """`types.MappingProxyType` of the underlying data."""
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
        """Return a `~collections.KeysView` of the collection."""
        return self._data.keys()

    def values(self) -> ValuesView[V]:
        """Return a `~collections.ValuesView` of the collection."""
        return self._data.values()

    def items(self) -> ItemsView[str, V]:
        """Return an `~collections.ItemsView` of the collection."""
        return self._data.items()

    @property
    def _k0(self) -> str:
        """The first key."""
        return next(iter(self._data.keys()))

    @property
    def _v0(self) -> V:
        """The first value."""
        return next(iter(self._data.values()))

    def __getattr__(self, key: str) -> MappingProxyType[str, Any]:
        """Map any unkown methods to the contained data."""
        # Have to special-case class-level property.
        if key in ("__isabstractmethod__",):
            return object.__getattribute__(self, key)
        # Check if the underlying data has the key, erroring if it doesn't.
        elif hasattr(self._v0, key):
            return MappingProxyType({k: getattr(v, key) for k, v in self.items()})
        raise AttributeError(f"{self.__class__.__name__!r} object has no attribute {key!r}")
