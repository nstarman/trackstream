from __future__ import annotations

# STDLIB
import inspect
from abc import ABCMeta, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, fields
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any, Callable, TypeVar

# THIRD PARTY
import astropy.coordinates as coords
import astropy.units as u
import numpy as np
import numpy.lib.recfunctions as rfn

# LOCAL
from trackstream.track.utils import is_structured
from trackstream.track.width.base import TO_FORMAT, WidthBase
from trackstream.utils.descriptors.classproperty import classproperty

if TYPE_CHECKING:
    # LOCAL
    from .interpolated import InterpolatedWidth


__all__ = ["BaseWidth"]

##############################################################################
# TYPING

T = TypeVar("T")
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
            return None

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
        return None

    @classproperty
    @abstractmethod
    def dimensions(cls) -> u.PhysicalType | None:
        """Physical type of the width (or `None`)."""
        return None

    # TODO! make a classproperty. But problematic with ABCMeta
    @property
    @abstractmethod
    def corresponding_width_types(cls) -> dict[u.PhysicalType, None | type[BaseWidth]]:
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
            raise ValueError("cannot interpolate; too short.")
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
    # I/O

    @singledispatchmethod
    @classmethod
    def from_format(cls, data: object):
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

    def __array_function__(
        self, func: Callable[..., Any], types: tuple[type, ...], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        """Interface with :mod:`numpy` functions."""
        # LOCAL
        from .interop import WB_FUNCS

        if func not in WB_FUNCS:
            return NotImplemented

        # Get NumPyInfo on function, given type of self
        finfo = WB_FUNCS[func](self)
        if not finfo.validate_types(types):
            return NotImplemented
        return finfo.func(*args, **kwargs)


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
        return LENGTH

    @classproperty
    @abstractmethod
    def corresponding_representation_type(cls) -> None | type[coords.BaseRepresentation]:
        return None

    @property
    @abstractmethod
    def corresponding_width_types(cls) -> dict[u.PhysicalType, None | type[BaseWidth]]:
        raise NotImplementedError


# ===================================================================
# Kinematic Space


@dataclass(frozen=True)
class KinematicSpaceWidth(BaseWidth):
    """Width in velocity space."""

    @classproperty
    def dimensions(cls) -> u.PhysicalType:
        return SPEED

    @classproperty
    # @abstractmethod
    def corresponding_representation_type(cls) -> None | type[coords.BaseDifferential]:
        return None

    @property
    @abstractmethod
    def corresponding_width_types(self) -> dict[u.PhysicalType, type[WidthBase]]:
        raise NotImplementedError


##############################################################################


@TO_FORMAT.implements(np.ndarray, dispatch_on=BaseWidth)
def _to_format_ndarray(data, *args):
    return np.array(data, *args)


@TO_FORMAT.implements(u.Quantity, dispatch_on=BaseWidth)
def _to_format_quantity(data, *args):
    unit = u.StructuredUnit(tuple(getattr(data, f.name).unit for f in fields(data)))
    out = u.Quantity(np.array(data, *args), unit=unit)
    return out
