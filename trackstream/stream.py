# -*- coding: utf-8 -*-

"""Core Functions."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import copy
import re
from abc import ABCMeta, abstractmethod
from functools import cached_property
from typing import Any, Dict, Optional, Tuple, TypeVar, Union, cast

# THIRD PARTY
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import BaseCoordinateFrame, SkyCoord, UnitSphericalRepresentation
from astropy.table import Column, QTable, Table
from astropy.units import Quantity
from astropy.utils.decorators import lazyproperty
from astropy.utils.misc import indent
from astropy.visualization import quantity_support

# LOCAL
from trackstream._type_hints import FrameLikeType
from trackstream.core import StreamTrack, TrackStream
from trackstream.utils.coord_utils import resolve_framelike
from trackstream.utils.descriptors import InstanceDescriptor
from trackstream.utils.path import path_moments
from trackstream.utils.utils import abstract_attribute
from trackstream.visualization import CLike, StreamPlotDescriptorBase

__all__ = ["Stream"]

##############################################################################
# PARAMETERS

# Ensure Quantity is supported in plots
quantity_support()

# Error message for ABCs
_ABC_MSG = "Can't instantiate abstract class {} with abstract method {}"

# typing
S = TypeVar("S", bound="StreamBase")

##############################################################################
# CODE
##############################################################################


class StreamBasePlotDescriptor(StreamPlotDescriptorBase[S]):
    def in_icrs_frame(
        self,
        c: CLike = "tab:blue",
        plot_origin: bool = True,
        text_offset: float = 0,
        *,
        ax: Optional[plt.Axes] = None,
        format_ax: bool = True,
        **kwargs: Any,
    ) -> plt.Axes:
        stream, _ax = self._plot_setup(ax)

        super().in_icrs_frame(c=c, ax=_ax, format_ax=format_ax, **kwargs)

        if plot_origin:
            do = stream.origin.icrs
            self.plot_origin_label_lonlat(do.ra, do.dec, ax=_ax, text_offset=text_offset)

        return _ax

    def in_stream_frame(
        self,
        c: CLike = "tab:blue",
        plot_origin: bool = True,
        text_offset: float = 0,
        *,
        ax: Optional[plt.Axes] = None,
        format_ax: bool = True,
        **kwargs: Any,
    ) -> plt.Axes:
        stream, _ax = self._plot_setup(ax)

        super().in_stream_frame(c=c, ax=_ax, format_ax=format_ax, **kwargs)

        if plot_origin:
            do = stream.origin.transform_to(stream.frame)
            self.plot_origin_label_lonlat(do.lon, do.lat, ax=_ax, text_offset=text_offset)

        return _ax


class StreamBase(metaclass=ABCMeta):
    """Abstract base class for streams.

    Streams must define the following attributes / properties.

    Attributes
    ----------
    _data_max_lines : int
    data : BaseCoordinateFrame
    frame : BaseCoordinateFrame
    name : str
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
        raise TypeError(_ABC_MSG.format(self.__class__.__qualname__, "name"))

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
        raise TypeError(_ABC_MSG.format(self.__class__.__qualname__, "name"))

    @property
    def has_distances(self) -> bool:
        """Return `True` if ``.coords`` has distance information."""
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
        origin: str = repr(self.origin)
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


# ===================================================================


class StreamArmDescriptor(InstanceDescriptor["Stream"], StreamBase):
    """Descriptor on a `Stream` to have substreams describing a stream arm.

    This is an instance-level descriptor, so most attributes / methods point to
    corresponding methods on the parent instance.

    Attributes
    ----------
    full_name : str
        Full name of the stream arm, including the parent name.
    has_data : bool
        Boolean of whether this arm has data.
    index : `astropy.table.Column`
        Boolean array of which stars in the parent table are in this arm.
    """

    @cached_property
    def name(self) -> str:
        attr_name = list(filter(None, re.split(r"(\d+)", self._parent_attr)))
        # e.g. arm1 -> ["arm", "1"]
        return " ".join(attr_name)

    @cached_property
    def full_name(self) -> str:
        """Full name of the stream arm, including the parent name."""
        parent_name: str = pn if isinstance(pn := self._parent.name, str) else "Stream"
        name_parts: Tuple[str, str] = (parent_name, self.name)
        return ", ".join(name_parts)

    @property
    def index(self) -> Column:
        """Boolean array of which stars in the parent table are in this arm."""
        tailcolumn: Column = self._parent.data["tail"]
        return tailcolumn == self._parent_attr

    @cached_property
    def has_data(self) -> bool:
        """Boolean of whether this arm has data."""
        return any(self.index)

    @property
    def data(self) -> QTable:
        """Return subset of full stream table that is for this arm."""
        if not self.has_data:
            raise Exception(f"{self._parent_attr} has no data")
        return self._parent.data[self.index]

    @property
    def coords(self) -> SkyCoord:
        """The coordinates of the arm."""
        arm = cast(SkyCoord, self._parent.coords[self.index])
        return arm

    @cached_property
    def data_frame(self) -> BaseCoordinateFrame:
        return self._parent.data_frame

    @cached_property
    def frame(self) -> Optional[BaseCoordinateFrame]:
        return self._parent.frame

    @cached_property
    def origin(self) -> SkyCoord:
        return self._parent.origin

    @lazyproperty
    def _data_max_lines(self) -> int:
        data_max_lines = self._parent._data_max_lines
        return data_max_lines


# ===================================================================


class StreamPlotDescriptor(StreamBasePlotDescriptor["Stream"]):
    pass


class Stream(StreamBase):
    """A Stellar Stream.

    Parameters
    ----------
    data : `~astropy.table.Table`
        The stream data.
        If this has

    origin : `~astropy.coordinates.ICRS`
        The origin point of the stream (and rotated reference frame).

    data_err : `~astropy.table.QTable` (optional)
        The data_err must have (at least) column names
        ["x_err", "y_err", "z_err"]

    frame : `~astropy.coordinates.BaseCoordinateFrame` or None, optional keyword-only
        The stream frame. Locally linearizes the data.
        If None (default), need to fit for the frame.
    """

    arm1 = StreamArmDescriptor()
    arm2 = StreamArmDescriptor()

    plot = StreamPlotDescriptor()

    # ===============================================================

    _name: Optional[str]
    _origin: SkyCoord
    _system_frame: Optional[BaseCoordinateFrame]
    _cache: Dict[str, Any]
    _original_coord: Optional[SkyCoord]
    _data: QTable
    _data_max_lines: int
    _fitter: TrackStream

    def __init__(
        self,
        data: QTable,
        origin: BaseCoordinateFrame,
        data_err: Optional[Table] = None,
        *,
        frame: Optional[FrameLikeType] = None,
        name: Optional[str] = None,
        fitter: Optional[TrackStream] = None,
    ) -> None:
        self._name = name

        # system attributes
        self._origin = SkyCoord(origin, copy=False)
        self._system_frame = resolve_framelike(frame) if frame is not None else None
        # If system_frame is None it will have to be fit later

        self._cache = dict()  # TODO! improve

        # ---- Process the data ----
        # processed data

        self._original_coord = None  # set in _normalize_data
        self._data: QTable = self._normalize_data(data, data_err)
        self._data_max_lines: int = 10

        # ---- fitting the data ----
        # Make or check the tracker used to fit the data.
        # The fitter can be passed as an argument, but the default is to make
        # one here. The fitter is checked for on-sky vs 3D correctness.

        # Make, if not passed (default)
        if fitter is None:
            fitter = TrackStream(onsky=not self.has_distances)
        # If passed with no on-sky vs 3D preference, assign now.
        elif fitter.onsky is None:
            fitter.onsky = not self.has_distances
        # Check the fitter
        elif not fitter.onsky and not self.has_distances:
            raise ValueError(
                f"Cannot fit 3D track since Stream {self.full_name} does not have distances",
            )

        self._fitter = fitter

    # -----------------------------------------------------

    @property
    def name(self) -> Optional[str]:
        """The name of the stream."""
        return self._name

    @property
    def origin(self) -> SkyCoord:
        """Stream origin."""
        return self._origin.transform_to(self._best_frame)

    @cached_property
    def has_distances(self) -> bool:
        """Whether the data has distances or is on-sky"""
        data_onsky = issubclass(type(self.data_coords.data), UnitSphericalRepresentation)
        origin_onsky = issubclass(type(self._origin.data), UnitSphericalRepresentation)
        onsky: bool = data_onsky and origin_onsky
        return not onsky

    # ===============================================================

    @property
    def data(self) -> QTable:
        """Data `astropy.table.QTable`."""
        return self._data

    @property
    def data_coords(self) -> SkyCoord:
        """Get ``coord`` from data table."""
        return self.data["coord"]

    @cached_property
    def data_frame(self) -> BaseCoordinateFrame:
        """The `astropy.coordinates.BaseCoordinateFrame` of the data."""
        # Get the frame from the data
        frame: BaseCoordinateFrame = self.data_coords.frame.replicate_without_data()

        # Check if the frame's representation type should include distances
        if not self.has_distances:
            frame.representation_type = frame.representation_type._unit_representation

        return frame

    # ===============================================================

    @property
    def system_frame(self) -> Optional[BaseCoordinateFrame]:
        """Return a system-centric frame (or None).

        Determined from the argument ``frame`` at initialization.
        If None (default) and the method ``fit`` has been called,
        then a system frame has been found and cached.
        """
        frame: Optional[BaseCoordinateFrame]
        if self._system_frame is not None:
            frame = self._system_frame
        else:
            frame = self._cache.get("system_frame")  # Can be `None`

        return frame

    @property
    def frame(self) -> Optional[BaseCoordinateFrame]:
        """Alias for :attr:`Stream.system_frame`."""
        return self.system_frame

    @property
    def _best_frame(self) -> BaseCoordinateFrame:
        """:attr:`Stream.system_frame` unless its `None`, else :attr:`Stream.data_frame`."""
        frame = self.system_frame if self.system_frame is not None else self.data_frame
        return frame

    @lazyproperty
    def number_of_tails(self) -> int:
        """Number of tidal tails.

        Returns
        -------
        number_of_tails : int
            There can only be 1, or 2 tidal tails.
        """
        n: int = 2 if (self.arm1.has_data and self.arm2.has_data) else 1
        return n

    @property
    def coords(self) -> SkyCoord:
        """Data coordinates transformed to `Stream.system_frame`, if there is one."""
        return self.data_coords.transform_to(self._best_frame)

    # ===============================================================
    # Data normalization

    def _normalize_data(self, original: Table, original_err: Optional[Table]) -> QTable:
        """Normalize data table.

        Parameters
        ----------
        original : |Table|
        original_err : |Table| or None

        Returns
        -------
        data : :class:`~astropy.table.QTable`
        """
        data = QTable()  # going to be assigned in-place

        # 1) data probability. `data` modded in-place
        self._normalize_data_probability(original, out=data, default_weight=1)

        # 2) stream arm labels. `data` modded in-place
        self._normalize_data_arm(original, out=data)

        # 3) coordinates. `data` modded in-place
        self._normalize_data_coordinates(original, original_err, out=data)

        # 4) SOM ordering
        self._normalize_data_arm_index(original, out=data)

        # Metadata
        data.meta = copy.deepcopy(original.meta)

        return data

    def _normalize_data_probability(
        self, original: Table, *, out: QTable, default_weight: Union[float, u.Quantity] = 1.0
    ) -> None:
        """Data probability. Units of percent. Default is 100%.

        Parameters
        ----------
        original : |Table|
            The original data.

        out : |QTable|, keyword-only
            The normalized data.
        default_weight : float, optional keyword-only
            The default membership probability.
            If float, then range 0-1 maps to 0-100%.
            If has unit of percent, then unchanged

        Returns
        -------
        None
        """
        colns = [n.lower() for n in original.colnames]

        if "pmemb" in colns:
            index = colns.index("pmemb")
            oname = original.colnames[index]
            Pmemb = original[oname]
        else:
            Pmemb = np.ones(len(original)) * default_weight  # non-scalar

        out["Pmemb"] = u.Quantity(Pmemb).to(u.percent)  # in %

    def _normalize_data_arm(self, original: Table, *, out: QTable) -> None:
        """Parse the tail labels.

        Parameters
        ----------
        original : |Table|
            The original data.

        out : |QTable|
            The stream data.

        Returns
        -------
        None
        """
        # TODO!!! better
        out["tail"] = original["tail"]

        # group and add index
        out = out.group_by("tail")
        out.add_index("tail")

    def _normalize_data_coordinates(
        self, original: Table, original_err: Optional[Table] = None, *, out: QTable
    ) -> None:
        """Parse the data table.

        - the frame is stored in ``_data_frame``
        - the representation is stored in ``_data_rep``
        - the original data representation  is in ``_data``

        Parameters
        ----------
        original : |Table|
            The original data.
        original_err : |Table| or None, optional
            The error in the original data.

        out : |QTable|
            The stream data.

        Returns
        -------
        None
        """
        # ----------
        # 1) the data

        # First look for a column "coord"
        if "coord" in original.colnames:
            osc = SkyCoord(original["coord"], copy=False)
        else:
            osc = SkyCoord.guess_from_table(original)

        self._original_coord = osc
        osc_frame = osc.frame.replicate_without_data()
        osc_frame.representation_type = osc.representation_type

        # Convert frame and representation type
        frame = self.system_frame if self.system_frame is not None else osc_frame
        sc = osc.transform_to(frame)
        sc.representation_type = frame.representation_type

        # it's now clean and can be added
        out["coord"] = sc

        # Also store the components
        component_names = list(sc.get_representation_component_names("base").keys())

        # ----------
        # 2) the error
        # TODO! want the ability to convert errors into the frame of the data.
        # import gala.coordinates as gc
        # cov = np.array([[1, 0], [0, 1]])
        # gc.transform_pm_cov(sc.icrs, np.repeat(cov[None, :], len(sc), axis=0),
        #                     coord.Galactic())

        # the error is stored on either the original data table, or in a separate table.
        orig_err = original if original_err is None else original_err
        # Iterate over the components, getting the error
        n: str
        for n in component_names:
            dn: str = n + "_err"  # error component name
            # either get the error, or set it to zero.
            out[dn] = orig_err[dn] if dn in orig_err.colnames else 0 * getattr(sc, n)

    def _normalize_data_arm_index(self, original: Table, *, out: QTable) -> None:
        """Data probability. Units of percent. Default is 100%.

        Parameters
        ----------
        original : |Table|
            The original data.
        out : |QTable|, optional keyword-only
            The normalized data.

        Returns
        -------
        None
        """
        if "SOM" in original.colnames:
            out["SOM"] = original["SOM"]
        else:
            out["SOM"] = -1  # sentinel value

    # ===============================================================
    # Fitting

    def fit_frame(
        self, rot0: Optional[Quantity] = Quantity(0, u.deg), *, force: bool = False, **kwargs: Any
    ) -> BaseCoordinateFrame:
        """Fit a frame to the data.

        The frame is an on-sky rotated frame.
        To prevent a frame from being fit, the desired frame should be passed
        to the Stream constructor at initialization.

        Parameters
        ----------
        rot0 : |Quantity| or None.
            Initial guess for rotation.

        force : bool, optional keyword-only
            Whether to force a frame fit. Default `False`
        **kwargs : Any
            Passed to fitter. See Other Parameters for examples.

        Returns
        -------
        BaseCoordinateFrame
            The fit frame.

        Other Parameters
        ----------------
        bounds : array-like or None, optional
            Parameter bounds. If `None`, these are automatically constructed.
            If provided these are used over any other bounds-related arguments.
            ::
                [[rot_low, rot_up],
                 [lon_low, lon_up],
                 [lat_low, lat_up]]

        rot_lower, rot_upper : Quantity, optional keyword-only
            The lower and upper bounds in degrees.
            Default is (-180, 180] degree.
        origin_lim : Quantity, optional keyword-only
            The symmetric lower and upper bounds on origin in degrees.
            Default is 0.005 degree.

        fix_origin : bool, optional keyword-only
            Whether to fix the origin point. Default is False.
        leastsquares : bool, optional keyword-only
            Whether to to use :func:`~scipy.optimize.least_square` or
            :func:`~scipy.optimize.minimize`. Default is False.

        align_v : bool, optional keyword-only
            Whether to align by the velocity.

        Raises
        ------
        ValueError
            If a frame has already been fit and `force` is not `True`.
        TypeError
            If a system frame was given at initialization.
        """
        if self._system_frame is not None:
            raise TypeError("a system frame was given at initialization.")
        elif self.system_frame is not None and not force:
            raise ValueError("already fit. use ``force`` to re-fit.")

        frame, _ = self._fitter._fit_rotated_frame(self, rot0=rot0, **kwargs)
        # fitting with _fit_rotated_frame caches the frame, frame_fit, and
        # the fitter itself on `_fitter`
        # The frame is also cached here.
        self._cache["system_frame"] = frame

        return frame

    @property
    def track(self) -> StreamTrack:
        """Stream track.

        Raises
        ------
        ValueError
            If track is not fit.
        """
        track: Optional[StreamTrack] = self._cache.get("track")
        if track is None:
            raise ValueError("need to fit track.")
        return track

    def fit_track(
        self, *, force: bool = False, onsky: Optional[bool] = None, **kwargs: Any
    ) -> StreamTrack:
        """Make a stream track.

        Parameters
        ----------
        force : bool, optional keyword-only
            Whether to force a fit, even if already fit.

        on-sky : bool or None, optional keyword-only
            Should the track be fit by on-sky or with distances?
            If `None` (default) the data is inspected to see if it has distances.

        **kwargs
            Passed to :meth:`trackstream.TrackStream.fit`.

        Returns
        -------
        `trackstream.StreamTrack`

        Raises
        ------
        ValueError
        ----------
            If a frame has already been fit and `force` is not `True`.
        """
        if not force and "track" in self._cache:
            raise ValueError("already fit. use ``force`` to re-fit.")

        track: StreamTrack = self._fitter.fit(self, onsky=onsky, **kwargs)
        self._cache["track"] = track

        # Add SOM ordering to data
        self.data["SOM"] = np.empty(len(self.data), dtype=int)
        self.data["SOM"][self.arm1.index] = self._fitter._cache["arm1_visit_order"]
        if self.arm2.has_data:
            self.data["SOM"][self.arm2.index] = self._fitter._cache["arm2_visit_order"]

        return track

    # ===============================================================
    # Math on the Track (requires fitting track)

    def predict_track(
        self,
        affine: Optional[u.Quantity] = None,
        *,
        angular: bool = False,
    ) -> path_moments:
        """
        Parameters
        ----------
        affine : |Quantity| or None, optional
        angular : bool, optional keyword-only

        Returns
        -------
        `trackstream.utils.path.path_moments`
        """
        return self.track(affine=affine, angular=angular)

    # ===============================================================
    # Misc

    def _base_repr_(self, max_lines: Optional[int] = None) -> list:
        rs = super()._base_repr_(max_lines=max_lines)

        # 2) system frame
        system_frame: str = repr(self.system_frame)
        r = "  System Frame:"
        r += ("\n" + indent(system_frame)) if "\n" in system_frame else (" " + system_frame)
        rs[2] = r

        return rs
