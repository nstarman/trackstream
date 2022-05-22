# -*- coding: utf-8 -*-

"""Core Functions."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from abc import abstractmethod
from typing import Any, List, Optional, TypeVar, cast

# THIRD PARTY
from astropy.coordinates import (
    BaseCoordinateFrame,
    SkyCoord,
    UnitSphericalRepresentation,
)
from astropy.table import QTable
from astropy.utils.misc import indent
from astropy.visualization import quantity_support
from matplotlib.pyplot import Axes

# LOCAL
from trackstream.utils.misc import ABCwAMeta, abstract_attribute
from trackstream.visualization import DKindT, StreamPlotDescriptorBase

__all__: List[str] = []

##############################################################################
# PARAMETERS

# Ensure Quantity is supported in plots
quantity_support()

# Error message for ABCs
_ABC_MSG = "Can't instantiate abstract class {} with abstract method {}"

# typing
StreamBaseT = TypeVar("StreamBaseT", bound="StreamBase")

##############################################################################
# CODE
##############################################################################


class StreamBasePlotDescriptor(StreamPlotDescriptorBase[StreamBaseT]):
    """Plot methods for `trackstream.stream.base.StreamBase` objects."""

    def in_frame(
        self,
        frame: str = "ICRS",
        kind: DKindT = "positions",
        *,
        origin: bool = True,
        ax: Optional[Axes] = None,
        format_ax: bool = True,
        **kwargs: Any,
    ) -> Axes:
        """Plot stream in an |ICRS| frame.

        Parameters
        ----------
        frame : |Frame| or str, optional
            A frame instance or its name (a `str`, the default).
            Also supported is "stream", which is the stream frame
            of the enclosing instance.
        kind : {'positions', 'kinematics'}, optional
            The kind of plot.

        origin : bool, optional keyword-only
            Whether to plot the origin, by default `True`.

        ax : |Axes| or None, optional keyword-only
            Matplotlib |Axes|. `None` (default) uses the current axes
            (:func:`matplotlib.pyplot.gca`).
        format_ax : bool, optional keyword-only
            Whether to add the axes labels and info, by default `True`.
        **kwargs : Any
            Passed to :func:`matplotlib.pyplot.scatter`.

        Returns
        -------
        |Axes|
        """
        stream, _ax, *_ = self._setup(ax=ax)

        super().in_frame(frame=frame, kind=kind, ax=_ax, format_ax=format_ax, **kwargs)

        if origin:
            self.origin(stream.origin, frame=frame, kind=kind, ax=_ax, format_ax=format_ax)

        return _ax


class StreamBase(metaclass=ABCwAMeta):
    """Abstract base class for streams.

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

    plot: StreamBasePlotDescriptor = StreamBasePlotDescriptor()

    _data_max_lines: int = abstract_attribute()

    @property
    @abstractmethod
    def data(self) -> QTable:
        """The stream data."""
        raise TypeError(_ABC_MSG.format(self.__class__.__qualname__, "data"))

    @property
    @abstractmethod
    def data_frame(self) -> BaseCoordinateFrame:
        """The stream data."""
        raise TypeError(_ABC_MSG.format(self.__class__.__qualname__, "data"))

    @property
    @abstractmethod
    def coords(self) -> SkyCoord:
        """Coordinates."""
        raise TypeError(_ABC_MSG.format(self.__class__.__qualname__, "coords"))

    @property
    def coords_ord(self) -> SkyCoord:
        """The (ordered) coordinates of the arm."""
        return cast(SkyCoord, self.coords[self.data["order"]])

    @property
    @abstractmethod
    def frame(self) -> BaseCoordinateFrame:
        """The coordinate frame of the stream."""
        raise TypeError(_ABC_MSG.format(self.__class__.__qualname__, "frame"))

    @property
    @abstractmethod
    def name(self) -> Optional[str]:
        """The name of the stream."""
        raise TypeError(_ABC_MSG.format(self.__class__.__qualname__, "name"))

    @property
    @abstractmethod
    def origin(self) -> SkyCoord:
        """Origin in stream frame."""
        raise TypeError(_ABC_MSG.format(self.__class__.__qualname__, "origin"))

    @property
    def has_distances(self) -> bool:
        """Return `True` if ``.coords`` has distance information."""
        # TODO! better check
        return not issubclass(self.coords.data.__class__, UnitSphericalRepresentation)

    @property
    def full_name(self) -> Optional[str]:
        """The name of the stream."""
        return self.name

    # ===============================================================

    def _base_repr_(self, max_lines: Optional[int] = None) -> list:
        rs = []

        # 0) header (standard repr)
        header: str = object.__repr__(self)
        rs.append(header)

        # 1) name
        name = str(self.full_name)
        rs.append("  Name: " + name)

        # 2) frame
        frame: str = repr(self.frame)
        r = "  Frame:"
        r += ("\n" + indent(frame)) if "\n" in frame else (" " + frame)
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

        # 5) data table
        datarep: str = self.data._base_repr_(html=False, max_width=None, max_lines=max_lines)
        table: str = "\n\t".join(datarep.split("\n")[1:])
        rs.append("  Data:\n\t" + table)

        return rs

    def __repr__(self) -> str:
        s: str = "\n".join(self._base_repr_(max_lines=self._data_max_lines))
        return s

    def __len__(self) -> int:
        return len(self.data)