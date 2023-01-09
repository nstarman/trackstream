"""Width."""

from __future__ import annotations

# STDLIB
import copy as pycopy
import inspect
from abc import ABCMeta, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from functools import singledispatchmethod
from types import NotImplementedType
from typing import TYPE_CHECKING, Any, TypeVar

# THIRD PARTY
import astropy.coordinates as coords
import astropy.units as u
import numpy as np
import numpy.lib.recfunctions as rfn
from numpy.typing import NDArray

# LOCAL
from trackstream.track.utils import is_structured
from trackstream.track.width.base import FMT_OVERLOADS, WidthBase
from trackstream.utils.descriptors.classproperty import classproperty

if TYPE_CHECKING:
    # LOCAL
    from trackstream.track.width.interpolated import InterpolatedWidth


__all__ = ["BaseWidth"]

##############################################################################
# TYPING

Self = TypeVar("Self", bound="BaseWidth")
W1 = TypeVar("W1", bound="BaseWidth")
W2 = TypeVar("W2", bound="BaseWidth")


##############################################################################
# PARAMETERS

BASEWIDTH_KIND: dict[type[BaseWidth], u.PhysicalType] = {}

BASEWIDTH_REP: dict[type[coords.BaseRepresentationOrDifferential], type[BaseWidth]] = {}
# map Representation -> BaseWidth

LENGTH = u.get_physical_type("length")
ANGLE = u.get_physical_type("angle")
SPEED = u.get_physical_type("speed")
ANGULAR_SPEED = u.get_physical_type("angular speed")


##############################################################################
# CODE
##############################################################################


@dataclass(frozen=True)
class BaseWidth(WidthBase, metaclass=ABCMeta):
    """Base class for widths: vectors not implicitly defined from the origin.

    Subclasses define fields for each coordinate direction. Subclasses must
    define a class-property ``corresponding_representation_type`` that returns a
    :class:`astropy.coordinates.BaseRepresentation` sublclass instance or
    `None`. If it is not None, then the class is registered in
    ``BASEWIDTH_REP``. Subclasses must define a class-property ``dimensions``
    that returns the physical type, e.g. 'length' for configuration spaces,
    'speed' for velocity space.
    """

    def __init_subclass__(cls) -> None:
        if inspect.isabstract(cls):  # skip ABCs
            return

        # Register class.
        if cls.corresponding_representation_type is not None:
            BASEWIDTH_REP[cls.corresponding_representation_type] = cls

        if cls.dimensions is not None:
            BASEWIDTH_KIND[cls] = cls.dimensions

    # ===============================================================

    @classproperty  # TODO! this obscures inspect.isabstract
    @abstractmethod
    def corresponding_representation_type(cls) -> None | type[coords.BaseRepresentationOrDifferential]:
        """Representation type corresponding to the width type."""
        return

    @classproperty
    @abstractmethod
    def dimensions(cls) -> u.PhysicalType | None:
        """Physical type of the width (or `None`)."""
        return

    # TODO! make a classproperty. But problematic with ABCMeta
    @property
    @abstractmethod
    def corresponding_width_types(cls) -> dict[u.PhysicalType, None | type[BaseWidth]]:
        """Return a dictionary of corresponding width types."""
        raise NotImplementedError

    @property
    def units(self) -> u.StructuredUnit:
        """Return structured unit from fields."""
        ns = tuple(f.name for f in fields(self))
        return u.StructuredUnit(tuple(getattr(self, n).unit for n in ns), names=ns)

    # ===============================================================

    def interpolated(self: WidthBase, affine: u.Quantity) -> InterpolatedWidth:
        """Return affine-interpolated width.

        Parameters
        ----------
        affine : Quantity
            Strictly ascending parameter to interpolate the width.

        Returns
        -------
        `trackstream.track.width.InterpolatedWidth`
            Same width instance, interpolated by the affine parameter.
        """
        if len(self) < 2:
            msg = "cannot interpolate; too short."
            raise ValueError(msg)
        # LOCAL
        from trackstream.track.width.interpolated import InterpolatedWidth

        return InterpolatedWidth.from_format(self, affine=affine)

    def represent_as(self, width_type: type[W1], point: coords.BaseRepresentation) -> W1:
        """Transform the width to another representation type.

        Parameters
        ----------
        width_type : `trackstream.track.width.BaseWidth` subclass class
            The width type to which to transform this width.
        point : `astropy.coordinates.BaseRepresentation` instance
            The base of the vector from which this width is defined.

        Returns
        -------
        `trackstream.track.width.BaseWidth` subclass instance
            This width, transformed to type ``width_type``.
        """
        # LOCAL
        from trackstream.track.width.transforms import represent_as

        return represent_as(self, width_type, point)

    # ===============================================================
    # Magic Methods

    @singledispatchmethod
    def __lt__(self, other: object) -> Any:  # noqa: ARG002
        # see https://github.com/python/mypy/issues/11727 for returning Any
        return NotImplemented

    @singledispatchmethod
    def __getitem__(self: Self, key: object) -> Self:
        return replace(self, **{f.name: getattr(self, f.name)[key] for f in fields(self)})

    @singledispatchmethod
    def __setitem__(self, key: object, value: Any) -> Any:  # noqa: ARG002
        # see https://github.com/python/mypy/issues/11727 for returning Any
        msg = "not dispatched"
        raise NotImplementedError(msg)

    def __deepcopy__(self: W1, memo: dict[Any, Any]) -> W1:
        return type(self)(**{f.name: pycopy.deepcopy(getattr(self, f.name), memo=memo) for f in fields(self)})

    # ===============================================================
    # I/O

    @singledispatchmethod
    @classmethod
    def from_format(cls: type[Self], data: object) -> Self:
        """Construct this width instance from data of given type."""
        return super().from_format(data)

    @from_format.register(np.void)
    @from_format.register(np.ndarray)
    @classmethod
    def _from_format_structuredarray(cls, data: np.ndarray) -> BaseWidth:
        # From a structured array.
        if not is_structured(data):
            raise ValueError
        names = {f.name for f in fields(cls)}
        return cls(**{k: data[k] for k in data.dtype.names if k in names})

    @from_format.register(Mapping)
    @classmethod
    def _from_format_mapping(cls, data: Mapping[str, Any]) -> BaseWidth:
        # From a Mapping.
        names = (f.name for f in fields(cls))
        return cls(**{k: data[k] for k in names})

    # ===============================================================
    # Interoperability

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        """Interface with ``np.array``."""
        dt = np.dtype([(f.name, dtype) for f in fields(self)])
        x = np.c_[tuple(getattr(self, f.name).value for f in fields(self))]
        return rfn.unstructured_to_structured(x, dtype=dt)

    @__setitem__.register(Mapping)
    def _setitem_mapping(self, key: Mapping[str, Any], value: BaseWidth) -> None | NotImplementedType:
        # Setitem mut be implemented for each field.
        if key.keys() != {f.name for f in fields(self)}:
            raise ValueError
        elif key.keys() != {f.name for f in fields(value)}:
            raise ValueError

        # Delegate to contained fields
        for f in fields(self):
            k = f.name
            getattr(self, k)[key[k]] = getattr(value, k)

    @__getitem__.register(Mapping)
    def _getitem_mapping(self, key: Mapping[str, NDArray[np.int_]]) -> Any | NotImplementedType:
        return {f.name: getattr(self, f.name)[key[f.name]] for f in fields(self) if f.name in key}


@BaseWidth.__lt__.register(BaseWidth)
def _lt_basewidth(self, other: BaseWidth) -> dict[str, np.ndarray]:
    if not isinstance(other, self.__class__):
        return NotImplemented

    return {f.name: getattr(self, f.name) < getattr(other, f.name) for f in fields(self)}


# ===================================================================
# Configuration Space


@dataclass(frozen=True)
class ConfigSpaceWidth(BaseWidth, metaclass=ABCMeta):
    """Width in configuration space.

    Subclasses must define class-properties ``corresponding_width_types`` and
    ``corresponding_representation_type``.
    """

    @classproperty
    def dimensions(cls) -> u.PhysicalType:
        """Physical type of the width."""
        return LENGTH

    @classproperty
    @abstractmethod
    def corresponding_representation_type(cls) -> None | type[coords.BaseRepresentation]:
        """Representation type corresponding to the width type."""
        return

    @property
    @abstractmethod
    def corresponding_width_types(cls) -> dict[u.PhysicalType, None | type[BaseWidth]]:
        """The width types corresponding to this width type."""
        raise NotImplementedError


# ===================================================================
# Kinematic Space


@dataclass(frozen=True)
class KinematicSpaceWidth(BaseWidth):
    """Width in velocity space."""

    @classproperty
    def dimensions(cls) -> u.PhysicalType:
        """Physical type of the width."""
        return SPEED

    @classproperty
    def corresponding_representation_type(cls) -> None | type[coords.BaseDifferential]:
        """Representation type corresponding to the width type."""
        return

    @property
    @abstractmethod
    def corresponding_width_types(self) -> dict[u.PhysicalType, type[WidthBase]]:
        """The width types corresponding to this width type."""
        raise NotImplementedError


##############################################################################


@FMT_OVERLOADS.implements(to_format=np.ndarray, from_format=BaseWidth)
def _to_format_ndarray(cls: type[BaseWidth], data: BaseWidth, *args: Any) -> BaseWidth:  # noqa: ARG001
    return np.array(data, *args)


@FMT_OVERLOADS.implements(to_format=u.Quantity, from_format=BaseWidth)
def _to_format_quantity(cls: type[BaseWidth], data: BaseWidth, *args: Any) -> BaseWidth:
    unit = u.StructuredUnit(tuple(getattr(data, f.name).unit for f in fields(data)))
    return cls(np.array(data, *args), unit=unit)
