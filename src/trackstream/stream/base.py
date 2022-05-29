# -*- coding: utf-8 -*-

"""Core Functions."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from abc import ABCMeta, abstractmethod
from types import MappingProxyType
from typing import List, Optional, cast

# THIRD PARTY
from astropy.coordinates import BaseCoordinateFrame, SkyCoord, SphericalRepresentation
from astropy.table import QTable
from astropy.utils.misc import indent
from astropy.visualization import quantity_support
from attrs import define, field

# LOCAL
from .visualization import StreamBasePlotDescriptor
from trackstream.utils._attrs import _cache_proxy_factory, convert_if_none

__all__: List[str] = []

##############################################################################
# PARAMETERS

# Ensure Quantity is supported in plots
quantity_support()

# Error message for ABCs
_ABC_MSG = "Can't instantiate abstract class {} with abstract method {}"

##############################################################################
# CODE
##############################################################################


@define(frozen=True, slots=False, repr=False)
class StreamBase(metaclass=ABCMeta):
    """Abstract base class for streams, including stream arms.

    Streams must define the following attributes / properties.

    Attributes
    ----------
    data : `astropy.table.QTable`
        The Stream data.
    frame : `astropy.coordinates.BaseCoordinateFrame`
        The frame of the stream.
    name : str
        Name of the stream.
    """

    plot = StreamBasePlotDescriptor()

    # ===============================================================

    name: Optional[str] = field(default=None, kw_only=True)

    _cache: dict = field(kw_only=True, factory=dict, converter=convert_if_none(dict, deepcopy=True))
    cache: MappingProxyType = field(init=False, default=_cache_proxy_factory)
    _data_max_lines: int = field(init=False, default=10, kw_only=True)

    # ===============================================================

    @property
    @abstractmethod
    def data(self) -> QTable:
        """The stream data."""

    @property
    @abstractmethod
    def origin(self) -> SkyCoord:
        """The origin on the stream."""

    @property
    @abstractmethod
    def data_frame(self) -> BaseCoordinateFrame:
        """The frame of the coordinates in ``data``."""
        raise TypeError(_ABC_MSG.format(self.__class__.__qualname__, "data_frame"))

    @property
    @abstractmethod
    def data_coords(self) -> SkyCoord:
        raise TypeError(_ABC_MSG.format(self.__class__.__qualname__, "data_coords"))

    @property
    @abstractmethod
    def system_frame(self) -> Optional[BaseCoordinateFrame]:
        """The stream data."""
        raise TypeError(_ABC_MSG.format(self.__class__.__qualname__, "system_frame"))

    @property
    @abstractmethod
    def coords(self) -> SkyCoord:
        """Coordinates."""
        raise TypeError(_ABC_MSG.format(self.__class__.__qualname__, "coords"))

    @property
    @abstractmethod
    def coords_ord(self) -> SkyCoord:
        """The (ordered) coordinates. Requires fitting."""
        # TODO! total order
        return cast(SkyCoord, self.coords[self.data["order"]])

    @property
    def has_distances(self) -> bool:
        """Return `True` if ``.coords`` has distance information."""
        return (
            self.coords.represent_as(SphericalRepresentation).distance.unit.physical_type
            == "length"
        )

    @property
    def has_kinematics(self) -> bool:
        """Return `True` if ``.coords`` has distance information."""
        return "s" in self.coords.data.differentials

    @property
    def full_name(self) -> Optional[str]:
        """The name of the stream."""
        return self.name

    # ===============================================================

    def __base_repr__(self, max_lines: Optional[int] = None) -> list:
        rs = []

        # 0) header (standard repr)
        header: str = object.__repr__(self)
        rs.append(header)

        # 1) name
        name = str(self.full_name)
        rs.append("  Name: " + name)

        # 2) frame
        system_frame: str = repr(self.system_frame)
        r = "  System Frame:"
        r += ("\n" + indent(system_frame)) if "\n" in system_frame else (" " + system_frame)
        rs.append(r)

        # 3) Origin
        origin: str = repr(self.origin.transform_to(self.data_frame))
        r = "  Origin:"
        r += ("\n" + indent(origin)) if "\n" in origin else (" " + origin)
        rs.append(r)

        # 4) data frame
        data_frame: str = repr(self.data_frame)
        r = "  Data Frame:"
        r += ("\n" + indent(data_frame)) if "\n" in data_frame else (" " + data_frame)
        rs.append(r)

        return rs

    def __repr__(self) -> str:
        s: str = "\n".join(self.__base_repr__(max_lines=self._data_max_lines))
        return s

    def __len__(self) -> int:
        return len(self.data)
