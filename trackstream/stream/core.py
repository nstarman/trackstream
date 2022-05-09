# -*- coding: utf-8 -*-

"""Core Functions."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import copy
from functools import cached_property
from typing import TYPE_CHECKING, Any, Optional, TypedDict, cast

# THIRD PARTY
import astropy.units as u
from astropy.coordinates import BaseCoordinateFrame, SkyCoord, UnitSphericalRepresentation
from astropy.table import QTable, Table
from astropy.units import Quantity
from astropy.utils.misc import indent
from numpy import arange, concatenate, empty, nonzero, zeros

# LOCAL
from .arm import StreamArmDescriptor
from .base import StreamBase, StreamBasePlotDescriptor
from .utils import StreamDataNormalizer
from trackstream._type_hints import FrameLikeType
from trackstream.track import StreamTrack, TrackStream
from trackstream.utils.coord_utils import resolve_framelike

if TYPE_CHECKING:
    # LOCAL
    from trackstream.track.path import path_moments

__all__ = ["Stream"]

##############################################################################
# PARAMETERS

# Error message for ABCs
_ABC_MSG = "Can't instantiate abstract class {} with abstract method {}"


class _StreamCache(TypedDict):
    """Cache for Stream."""

    system_frame: Optional[BaseCoordinateFrame]
    track: Optional[StreamTrack]


##############################################################################
# CODE
##############################################################################


class Stream(StreamBase):
    """A Stellar Stream.

    Parameters
    ----------
    data : `~astropy.table.Table`
        The stream data.

    origin : `~astropy.coordinates.ICRS`
        The origin point of the stream (and rotated reference frame).

    data_err : `~astropy.table.QTable` (optional)
        The data_err must have (at least) column names
        ["x_err", "y_err", "z_err"]

    frame : `~astropy.coordinates.BaseCoordinateFrame` or None, optional keyword-only
        The stream frame. Locally linearizes the data.
        If `None` (default), need to fit for the frame.

    name : str or None, optional keyword-only
        The name fo the stream.

    fitter : TrackStream or None, optional keyword-only
        Fitter for the track.
    """

    _Normalizer = StreamDataNormalizer["Stream"]()

    arm1 = StreamArmDescriptor()
    arm2 = StreamArmDescriptor()

    plot = StreamBasePlotDescriptor["Stream"]()

    # ===============================================================

    _data: QTable
    _original_coord: Optional[SkyCoord]
    _origin: SkyCoord
    _name: Optional[str]
    _init_system_frame: Optional[BaseCoordinateFrame]
    _fitter: TrackStream
    _data_max_lines: int
    _cache: _StreamCache

    def __init__(
        self,
        data: QTable,
        origin: SkyCoord,
        data_err: Optional[Table] = None,
        *,
        frame: Optional[FrameLikeType] = None,
        name: Optional[str] = None,
        fitter: Optional[TrackStream] = None,
    ) -> None:
        self._name = name

        # system attributes
        self._origin = SkyCoord(origin, copy=False)
        self._init_system_frame = resolve_framelike(frame) if frame is not None else None
        # If system_frame is None it will have to be fit later

        self._cache = _StreamCache.fromkeys(_StreamCache.__required_keys__)  # type: ignore

        # ---- Process the data ----
        # processed data

        self._original_coord = None  # set in _normalize_data
        self._data = self._Normalizer.run(data, data_err)
        self._data_max_lines = 10

        # ---- fitting the data ----
        # Make or check the tracker used to fit the data.
        # The fitter can be passed as an argument, but the default is to make
        # one here. The fitter is checked for on-sky vs 3D correctness.

        # Make, if not passed (default)
        if fitter is None:
            fitter = TrackStream(onsky=not self.has_distances)
        # If passed with no on-sky vs 3D preference, assign now.
        elif fitter.onsky is None:
            fitter = copy.deepcopy(fitter)
            fitter.onsky = not self.has_distances
        # Check the fitter
        elif not fitter.onsky and not self.has_distances:
            raise ValueError(
                f"Cannot fit 3D track since Stream {self.full_name} does not have distances",
            )
        else:
            fitter = copy.deepcopy(fitter)

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
        # TODO a more robust check
        data_onsky = issubclass(type(self.data_coords.data), UnitSphericalRepresentation)
        origin_onsky = issubclass(type(self._origin.data), UnitSphericalRepresentation)
        onsky: bool = data_onsky and origin_onsky
        return not onsky

    @property
    def has_kinematics(self) -> bool:
        return "s" in self.data_coords.data.differentials

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
        if self._init_system_frame is not None:
            frame = self._init_system_frame
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

    @cached_property
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
        """Data coordinates transformed to `Stream.system_frame` (if there is one)."""
        frame = self._best_frame

        c = self.data_coords.transform_to(frame)
        c.representation_type = frame.representation_type
        c.differential_type = frame.differential_type

        return c

    @property
    def coords_ord(self) -> SkyCoord:
        """The (ordered) coordinates of the arm."""
        arm1 = nonzero(self.data["tail"] == "arm1")[0]
        order1 = arm1[self.data["order"][arm1]]  # order within arm1

        arm2 = nonzero(self.data["tail"] == "arm2")[0]
        order2 = arm2[self.data["order"][arm2]]

        order = concatenate((order1[::-1], order2))

        return cast(SkyCoord, self.coords[order])

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
            Whether to force a frame fit. Default `False`.
            Will only fit if a frame was not specified at initialization.

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
        if self._init_system_frame is not None:
            raise TypeError("a system frame was given at initialization.")
        elif self.system_frame is not None and not force:
            raise ValueError("already fit. use ``force`` to re-fit.")

        # Note on control flow: self -> _fitter(self) -> self
        frame, _ = self._fitter._fit_rotated_frame(self, rot0=rot0, **kwargs)
        # fitting with `_fit_rotated_frame` caches the frame, frame_fit, and
        # the fitter itself on `_fitter`.
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
            raise ValueError("need to fit track. See ``Stream.fit_track(...)``.")
        return track

    def fit_track(
        self,
        *,
        force: bool = False,
        onsky: Optional[bool] = None,
        kinematics: Optional[bool] = None,
        **kwargs: Any,
    ) -> StreamTrack:
        """Make a stream track.

        Parameters
        ----------
        force : bool, optional keyword-only
            Whether to force a fit, even if already fit.

        onsky : bool or None, optional keyword-only
            Should the track be fit on-sky or including distances? If `None`
            (default) the data is inspected to see if it has distances.
        kinematics : bool or None, optional keyword-only
            Should the track be fit with or without kinematics? If `None`
            (default) the data is inspected to see if it has kinematic
            information.

        **kwargs
            Passed to :meth:`trackstream.TrackStream.fit`.

        Returns
        -------
        `trackstream.StreamTrack`

        Raises
        ------
        ValueError
            If a frame has already been fit and ``force`` is not `True`.
        ValueError
            If ``onsky`` is False and the stream does not have distance
            information.
        """
        # Check if already fit
        if not force and "track" in self._cache:
            raise ValueError("already fit. use ``force`` to re-fit.")

        # Check if onsky will work
        if onsky is False and not self.has_distances:
            raise ValueError("stream does not have distance information and cannot be fit on-sky")
        if kinematics is True and not self.has_kinematics:
            raise ValueError(
                "stream does not have kinematic information and cannot be fit with kinematics"
            )

        track: StreamTrack = self._fitter.fit(
            self, onsky=onsky, kinematics=kinematics, force=force, **kwargs
        )
        self._cache["track"] = track

        # Add SOM ordering to data
        self.data["order"] = empty(len(self.data), dtype=int)
        self.data["order"][self.arm1.index] = self._fitter._cache["arm1_visit_order"]
        if self.arm2.has_data:
            self.data["order"][self.arm2.index] = self._fitter._cache["arm2_visit_order"]

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

        # 2) system frame replaces "Frame"
        system_frame: str = repr(self.system_frame)
        r = "  System Frame:"
        r += ("\n" + indent(system_frame)) if "\n" in system_frame else (" " + system_frame)
        rs[2] = r

        return rs
