# -*- coding: utf-8 -*-

"""Core Functions."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from functools import cached_property
from typing import TYPE_CHECKING, Any, Optional, TypedDict, cast

# THIRD PARTY
import astropy.units as u
from astropy.coordinates import (
    BaseCoordinateFrame,
    RadialDifferential,
    SkyCoord,
    UnitSphericalRepresentation,
)
from astropy.table import QTable, Table
from astropy.units import Quantity
from astropy.utils.misc import indent
from numpy import concatenate, empty, nonzero

# LOCAL
from .arm import StreamArmDescriptor
from .base import StreamBase, StreamBasePlotDescriptor
from .utils import StreamDataNormalizer
from trackstream._type_hints import CoordinateLikeType
from trackstream.track import StreamTrack, TrackStream
from trackstream.track.rotated_frame import FrameOptimizeResult, RotatedFrameFitter
from trackstream.utils.coord_utils import resolve_framelike

if TYPE_CHECKING:
    # LOCAL
    from trackstream.track.path import path_moments

__all__ = ["Stream"]

##############################################################################
# PARAMETERS


class _StreamCache(TypedDict):
    """Cache for Stream."""

    # frame
    system_frame: Optional[BaseCoordinateFrame]
    frame_fit: Optional[FrameOptimizeResult]
    frame_fitter: Optional[RotatedFrameFitter]
    # track
    track: Optional[StreamTrack]
    track_fitter: Optional[TrackStream]


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
    _data_max_lines: int
    _cache: _StreamCache

    def __init__(
        self,
        data: QTable,
        origin: SkyCoord,
        data_err: Optional[Table] = None,
        *,
        frame: Optional[CoordinateLikeType] = None,
        name: Optional[str] = None,
        # track: StreamTrack = None  # TODO!
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

    # -----------------------------------------------------

    @property
    def name(self) -> Optional[str]:
        """The name of the stream."""
        return self._name

    @property
    def origin(self) -> SkyCoord:
        """Stream origin."""
        # SkyCoord caches transformations, for speed.
        return self._origin.transform_to(self._best_frame)

    @cached_property
    def has_distances(self) -> bool:
        """Whether the data has distances or is on-sky"""
        data_onsky = self.data_coords.spherical.distance.unit.physical_type == "dimensionless"
        origin_onsky = self._origin.spherical.distance.unit.physical_type == "dimensionless"
        onsky: bool = data_onsky and origin_onsky
        return not onsky

    @cached_property
    def has_kinematics(self) -> bool:
        hasks = "s" in self.data_coords.data.differentials

        # For now can't do only radial diffs # TODO! ease restriction
        if hasks:
            hasks &= not isinstance(self.data_coords.data.differentials["s"], RadialDifferential)

        return hasks

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
        reptype = self.data_coords.frame.representation_type
        if not self.has_distances:
            reptype = getattr(reptype, "_unit_representation", reptype)

        # Get the frame from the data
        frame: BaseCoordinateFrame = self.data_coords.frame.replicate_without_data(
            representation_type=reptype,
        )

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
        n: int = self.arm1.has_data + self.arm2.has_data  # bools add to int
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
        self,
        fitter: Optional[RotatedFrameFitter] = None,
        rot0: Optional[Quantity] = Quantity(0, u.deg),
        *,
        force: bool = False,
        **kwargs: Any,
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
            raise ValueError("a frame has already beem fit. use ``force`` to re-fit.")

        if fitter is None:
            fitter = RotatedFrameFitter(
                origin=self.origin,
                frame=self.data_frame,
                representation_type=UnitSphericalRepresentation,
                # TODO! the kwargs
            )

        fitted = fitter.fit(self.data_coords, rot0=rot0, **kwargs)

        # cache
        self._cache["frame_fitter"] = fitter
        self._cache["frame_fit"] = fitted
        self._cache["system_frame"] = fitted.frame

        return fitted.frame

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
        fitter: Optional[TrackStream] = None,
        *,
        force: bool = False,
        # onsky: Optional[bool] = None,
        # kinematics: Optional[bool] = None,
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
        if not force and self._cache["track"] is not None:
            raise ValueError("already fit. use ``force`` to re-fit.")

        if fitter is None:
            fitter = TrackStream(onsky=not self.has_distances, kinematics=self.has_kinematics)

        track: StreamTrack = fitter.fit(self, force=force, **kwargs)

        # TODO! this caching in `fitter.fit`
        self._cache["track_fitter"] = fitter
        self._cache["track"] = track

        # Add SOM ordering to data
        self.data["order"] = empty(len(self.data), dtype=int)
        self.data["order"][self.arm1.index] = fitter._cache["arm1_visit_order"]
        if self.arm2.has_data:
            self.data["order"][self.arm2.index] = fitter._cache["arm2_visit_order"]

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
