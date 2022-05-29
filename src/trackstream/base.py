# -*- coding: utf-8 -*-

# STDLIB
from abc import ABCMeta
from types import MappingProxyType
from typing import (
    Dict,
    Iterator,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    cast,
)

# THIRD PARTY
from astropy.coordinates import (
    BaseCoordinateFrame,
    BaseDifferential,
    BaseRepresentation,
)
from attr import Attribute
from attrs import define, field

# LOCAL
from trackstream.utils import resolve_framelike
from trackstream.utils._attrs import attrs_factory_decorator

__all__ = ["FramedBase", "CollectionBase"]


CollectionBaseT = TypeVar("CollectionBaseT", bound="CollectionBase")
K = TypeVar("K")
V = TypeVar("V")


class _HasFrame(Protocol):
    @property
    def frame(self) -> BaseCoordinateFrame:
        ...


@attrs_factory_decorator
def frame_representation_type_factory(self: _HasFrame) -> Type[BaseRepresentation]:
    return self.frame.representation_type


@attrs_factory_decorator
def frame_differential_type_factory(self: _HasFrame) -> Type[BaseDifferential]:
    return self.frame.differential_type


@define(frozen=True, kw_only=True, slots=False)
class FramedBase(metaclass=ABCMeta):
    """Base class for most objects. Provides a frame and representation type.

    Parameters
    ----------
    frame : `~astropy.coordinates.BaseCoordinateFrame`, keyword-only
        The frame.
    representation_type : `astropy.coordinates.BaseRepresentation` or None, optional keyword-only
        The representation type for the `frame`. If `None` (default) uses
        the current representation type of the `frame`.
    """

    frame: BaseCoordinateFrame = field(converter=resolve_framelike)
    frame_representation_type: Type[BaseRepresentation] = field()
    frame_differential_type: Optional[Type[BaseDifferential]] = field()

    @frame_representation_type.default  # type: ignore
    def _frame_representation_type_factory(self) -> Type[BaseRepresentation]:
        return frame_representation_type_factory.factory(self)

    @frame_differential_type.default  # type: ignore
    def _frame_differential_type_factory(self) -> Type[BaseDifferential]:
        return frame_differential_type_factory.factory(self)

    @frame_representation_type.validator  # type: ignore
    def _frame_representation_type_validator(
        self, _: Attribute, value: Optional[Type[BaseRepresentation]]
    ):
        if value is None:
            object.__setattr__(
                self, "frame_representation_type", frame_representation_type_factory.factory(self)
            )

    @frame_differential_type.validator  # type: ignore
    def _frame_differential_type_validator(
        self, _: Attribute, value: Optional[Type[BaseDifferential]]
    ):
        if value is None:
            object.__setattr__(
                self, "frame_differential_type", frame_differential_type_factory.factory(self)
            )

    def __attrs_post_init__(self):
        frame = self.frame.replicate_without_data(
            representation_type=self.frame_representation_type,
            differential_type=self.frame_differential_type,
        )
        object.__setattr__(self, "frame", frame)

    # ---------------------------

    @property
    def _rep_attrs(self) -> Tuple[str, ...]:
        attrs = tuple(getattr(self.frame_representation_type, "attr_classes", {}).keys())
        return cast(Tuple[str, ...], attrs)

    @property
    def _dif_attrs(self) -> Tuple[str, ...]:
        attrs = tuple(getattr(self.frame_differential_type, "attr_classes", {}).keys())
        return cast(Tuple[str, ...], attrs)


##############################################################################


@define(frozen=True, repr=False, slots=False)
class CollectionBase(Mapping[str, V]):

    _data: dict = field(init=True, factory=dict, converter=dict)
    name: Optional[str] = field(default=None, kw_only=True)

    __dataview__: MappingProxyType = field(init=False, default=None)

    def __init__(
        self, data: Optional[Dict[str, V]] = None, /, name: Optional[str] = None, **kwargs: V
    ) -> None:
        d = dict(data) if data is not None else {}
        d.update(kwargs)
        self.__attrs_init__(data=d, name=name)  # type: ignore

    @property
    def _dataview(self) -> MappingProxyType:
        if self.__dataview__ is None:
            object.__setattr__(self, "__dataview__", MappingProxyType(self._data))
        return self.__dataview__

    def __getitem__(self, key: str) -> V:
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
