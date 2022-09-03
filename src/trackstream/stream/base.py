"""Core Functions."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

# THIRD PARTY
import astropy.units as u
from astropy.coordinates import BaseCoordinateFrame, RadialDifferential, SkyCoord
from astropy.utils.misc import indent
from matplotlib.axes import Axes

# LOCAL
from trackstream.stream.visualization import StreamPlotDescriptorBase
from trackstream.utils.descriptors.attribute import Attribute
from trackstream.utils.descriptors.cache import CacheProperty
from trackstream.utils.visualization import DKindT

if TYPE_CHECKING:
    # THIRD PARTY
    from astropy.table import QTable


__all__: list[str] = []

##############################################################################
# TYPING

StreamLikeT = TypeVar("StreamLikeT", bound="StreamLike")
StreamBaseT = TypeVar("StreamBaseT", bound="StreamBase")


class SupportsFrame(Protocol):
    @property
    def frame(self) -> BaseCoordinateFrame | None:
        ...


class StreamLike(Protocol):
    """Stream-like Protocol."""

    cache: CacheProperty
    flags: Any

    @property
    def plot(self) -> object:
        ...

    @property
    def full_name(self) -> str | None:
        ...

    @property
    def coords(self) -> SkyCoord:
        ...

    @property
    def frame(self) -> BaseCoordinateFrame | None:
        ...

    @property
    def origin(self) -> SkyCoord:
        ...


##############################################################################
# PARAMETER

FRAME_NONE_ERR = ValueError("frame is None; need to fit a frame")


_ABC_MSG = "Can't instantiate abstract class {} with abstract method {}"
# Error message for ABCs.
# ABCs only prevent a subclass from being defined if it doesn't override the
# necessary methods. ABCs do not prevent empty methods from being called. This
# message is for errors in abstract methods.


##############################################################################
# CODE
##############################################################################


@dataclass(frozen=True)
class Flags:
    minPmemb: u.Quantity = u.Quantity(80, unit=u.percent)
    table_repr_max_lines: int = 10

    def set(self, **kwargs: Any) -> None:
        """Set the value of a flag."""
        for key, value in kwargs.items():
            if not isinstance(value, type(getattr(self, key))):
                # TODO! allow for more general check
                raise TypeError

            object.__setattr__(self, key, value)


##############################################################################


@dataclass(frozen=True)
class StreamBasePlotDescriptor(StreamPlotDescriptorBase[StreamBaseT]):
    """Plot methods for `trackstream.stream.base.StreamBase` objects."""

    def in_frame(
        self,
        frame: str = "icrs",
        kind: DKindT = "positions",
        *,
        origin: bool = False,
        ax: Axes | None = None,
        format_ax: bool = False,
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
        _, _ax, *_ = self._setup(ax=ax)

        super().in_frame(frame=frame, kind=kind, ax=_ax, format_ax=format_ax, **kwargs)

        if origin:
            self.origin(frame=frame, kind=kind, ax=_ax, format_ax=format_ax)

        return _ax


##############################################################################


@dataclass(frozen=True)
class StreamBase:
    """Abstract base class for stream arms, and collections thereof.

    Streams must define the following attributes / properties.

    Attributes
    ----------
    data : `astropy.table.QTable`
        The Stream data.
    origin : `astropy.coordinates.SkyCoord`
        The origin of the stream.
    name : str
        Name of the stream.
    """

    cache = CacheProperty["StreamBase"]()
    flags = Attribute(Flags(minPmemb=u.Quantity(90, unit=u.percent), table_repr_max_lines=10), attrs_loc="__dict__")

    plot = StreamBasePlotDescriptor()

    # ===============================================================
    # Initializatino

    # TODO! py3.10 fixes the problems of ordering in subclasses
    # data: QTable
    # """The stream data table."""
    # name: str | None
    # """The name of the stream."""
    # prior_cache: InitVar[dict] | None = None
    # def __post_init__(self, prior_cache: dict | None) -> None:
    #     self.cache = prior_cache

    # this is included only for type hinting
    def __post_init__(self) -> None:
        self._cache: dict[str, Any]
        self.data: QTable
        self.origin: SkyCoord
        self.name: str | None

    # ===============================================================
    # Flags

    @property
    def has_distances(self) -> bool:
        """Whether the data has distances or is on-sky."""
        data_onsky = self.data["coords"].spherical.distance.unit.physical_type == "dimensionless"
        origin_onsky = self.origin.spherical.distance.unit.physical_type == "dimensionless"
        onsky: bool = data_onsky and origin_onsky
        return not onsky

    @property
    def has_kinematics(self) -> bool:
        """Return `True` if ``.coords`` has distance information."""
        has_vs = "s" in self.data["coords"].data.differentials

        # For now can't do RadialDifferential
        if has_vs:
            has_vs &= not isinstance(self.data["coords"].data.differentials["s"], RadialDifferential)

        return has_vs

    # ===============================================================

    # @property
    # @abstractmethod
    # def origin(self) -> SkyCoord:
    #     """The origin of the stream."""
    #     raise TypeError(_ABC_MSG.format(self.__class__.__qualname__, "origin"))

    @property
    # @abstractmethod
    def data_frame(self) -> BaseCoordinateFrame:
        """The frame of the coordinates in ``data``."""
        raise TypeError(_ABC_MSG.format(self.__class__.__qualname__, "data_frame"))

    @property
    # @abstractmethod
    def data_coords(self) -> SkyCoord:
        raise TypeError(_ABC_MSG.format(self.__class__.__qualname__, "data_coords"))

    @property
    # @abstractmethod
    def frame(self) -> BaseCoordinateFrame | None:
        """The stream data."""
        raise TypeError(_ABC_MSG.format(self.__class__.__qualname__, "frame"))

    @property
    # @abstractmethod
    def coords(self) -> SkyCoord:
        """Coordinates."""
        raise TypeError(_ABC_MSG.format(self.__class__.__qualname__, "coords"))

    @property
    def full_name(self) -> str | None:
        """The name of the stream."""
        return self.name

    # ===============================================================

    def __base_repr__(self, max_lines: int | None = None) -> list[str]:
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

        return rs

    def __repr__(self) -> str:
        s: str = "\n".join(self.__base_repr__(max_lines=self.flags.table_repr_max_lines))
        return s

    def __len__(self) -> int:
        return len(self.data)
