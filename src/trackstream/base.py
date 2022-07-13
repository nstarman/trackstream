from __future__ import annotations

# STDLIB
from abc import ABCMeta
from types import MappingProxyType
from typing import Iterator, Mapping, Type, TypeVar, ValuesView, cast

# THIRD PARTY
from astropy.coordinates import (
    BaseCoordinateFrame,
    BaseDifferential,
    BaseRepresentation,
)
from attrs import define, field

# LOCAL
from trackstream._typing import HasFrame
from trackstream.utils import resolve_framelike
from trackstream.utils._attrs import attrs_factory_decorator

__all__ = ["FramedBase", "CollectionBase"]


CollectionBaseT = TypeVar("CollectionBaseT", bound="CollectionBase")
K = TypeVar("K")
V = TypeVar("V")


@attrs_factory_decorator
def frame_representation_type_factory(self: HasFrame) -> type[BaseRepresentation]:
    return self.frame.representation_type


@attrs_factory_decorator
def frame_differential_type_factory(self: HasFrame) -> type[BaseDifferential]:
    return cast(Type[BaseDifferential], self.frame.differential_type)


@define(frozen=True, kw_only=True, slots=False)
class FramedBase(metaclass=ABCMeta):
    """Base class for objects with a fixed |Frame| and |Representation| type.

    Parameters
    ----------
    frame : |Frame|, keyword-only
        The coordinate frame.
    frame_representation_type : |Representation| or None, optional keyword-only
        The representation type for the |Frame|. If `None` (default) uses
        the current representation type of the |Frame|.
    frame_differential_type : |Representation| or None, optional keyword-only
        The differential type for the |Frame|. If `None` (default) uses
        the current differential type of the |Frame|.
    """

    frame: BaseCoordinateFrame = field(converter=resolve_framelike)
    frame_representation_type: type[BaseRepresentation] = field()
    frame_differential_type: type[BaseDifferential] | None = field()

    @frame_representation_type.default  # type: ignore
    def _frame_representation_type_default(self) -> type[BaseRepresentation]:
        """Generate default value for ``frame_representation_type``."""
        return frame_representation_type_factory.factory(self)

    @frame_differential_type.default  # type: ignore
    def _frame_differential_type_default(self) -> type[BaseDifferential]:
        """Generate default value for ``frame_differential_type``."""
        return frame_differential_type_factory.factory(self)

    def __attrs_post_init__(self):
        """Correct the rep/dif-type if passed `None`."""
        # Change default value for ``frame_representation_type``
        if self.frame_representation_type is None:
            reptype = frame_representation_type_factory.factory(self)
            object.__setattr__(self, "frame_representation_type", reptype)

        # Change default value for ``frame_differential_type``
        if self.frame_differential_type is None:
            diftype = frame_differential_type_factory.factory(self)
            object.__setattr__(self, "frame_differential_type", diftype)

        # Set frame with corrected rep/dif-type
        frame = self.frame.replicate_without_data(
            representation_type=self.frame_representation_type,
            differential_type=self.frame_differential_type,
        )
        object.__setattr__(self, "frame", frame)

    # ---------------------------

    @property
    def _frame_rep_attrs(self) -> ValuesView[str]:
        """Representation attributes names."""
        return self.frame.get_representation_component_names("base").values()

    @property
    def _frame_dif_attrs(self) -> ValuesView[str]:
        """Differential attributes names."""
        return self.frame.get_representation_component_names("s").values()


##############################################################################


@define(frozen=True, repr=False, slots=False)
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

    _data: dict = field(init=True, factory=dict, converter=dict)
    name: str | None = field(default=None, kw_only=True)

    _dataview_: MappingProxyType = field(init=False, default=None, repr=False)

    def __init__(self, data: dict[str, V] | None = None, /, *, name: str | None = None, **kwargs: V) -> None:
        d = dict(data) if data is not None else {}
        d.update(kwargs)
        self.__attrs_init__(data=d, name=name)  # type: ignore

    @property
    def _dataview(self) -> MappingProxyType:
        """Read-only view of the data."""
        if self._dataview_ is None:
            object.__setattr__(self, "_dataview_", MappingProxyType(self._data))
        return self._dataview_

    def __getitem__(self, key: str) -> V:
        """Get 'key' from the data."""
        return self._data[key]

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._data!r})"

    def keys(self):
        return self._dataview.keys()

    def values(self):
        return self._dataview.values()

    def items(self):
        return self._dataview.items()

    def __getattr__(self, key: str) -> MappingProxyType:
        """Map any unkown methods to the contained fields."""
        return MappingProxyType({k: getattr(v, key) for k, v in self.items()})
