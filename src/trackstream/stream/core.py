"""Stream arm classes.

Stream arms are descriptors on a `trackstrea.Stream` class.

"""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import copy
from functools import cached_property
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

# THIRD PARTY
import astropy.units as u
import numpy as np
from astropy.coordinates import (
    BaseCoordinateFrame,
    RadialDifferential,
    SkyCoord,
    UnitSphericalRepresentation,
)
from astropy.table import QTable
from astropy.units import Quantity
from attrs import define, field
from numpy import ndarray
from numpy.lib.recfunctions import structured_to_unstructured
from typing_extensions import TypedDict

# LOCAL
from .base import StreamBase
from .utils import StreamArmDataNormalizer
from trackstream._typing import CoordinateLikeType
from trackstream.stream.outlier_detection import (
    OUTLIER_DETECTOR_CLASSES,
    OutlierDetectorBase,
)
from trackstream.utils._attrs import (
    _cache_factory,
    _cache_proxy_factory,
    _drop_properties,
    convert_if_none,
)
from trackstream.utils.coord_utils import resolve_framelike

if TYPE_CHECKING:
    # LOCAL
    from trackstream.fit.fitter import FitterStreamArmTrack
    from trackstream.fit.rotated_frame import FrameOptimizeResult, RotatedFrameFitter
    from trackstream.fit.track import StreamArmTrack


__all__ = ["StreamArm"]


##############################################################################
# PARAMETERS and `attrs`


class _StreamArmCache(TypedDict):
    """Cache for Stream."""

    # frame
    system_frame: BaseCoordinateFrame | None
    frame_fit: FrameOptimizeResult | None
    frame_fitter: RotatedFrameFitter | None
    # track
    track: StreamArmTrack | None
    track_fitter: FitterStreamArmTrack | None


def _opt_resolve_framelike(frame: CoordinateLikeType | None):
    return None if frame is None else resolve_framelike(frame)


##############################################################################
# CODE
##############################################################################


@define(frozen=True, repr=False, field_transformer=_drop_properties)
class StreamArm(StreamBase):
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

    # ===============================================================

    data: QTable = field()
    origin: SkyCoord = field()
    _data_err: QTable | None = field(kw_only=True, default=None, repr=False)

    _system_frame: BaseCoordinateFrame | None = field(default=None, kw_only=True, converter=_opt_resolve_framelike)

    _cache: dict = field(
        kw_only=True,
        factory=_cache_factory(_StreamArmCache),
        converter=convert_if_none(_cache_factory(_StreamArmCache), deepcopy=True),
    )
    cache: MappingProxyType = field(init=False, default=_cache_proxy_factory)

    _Normalizer: StreamArmDataNormalizer = field(init=False, factory=StreamArmDataNormalizer, repr=False)
    """Data Normalizer. Should only be called once!"""

    def __attrs_post_init__(self):
        # Run data normalizer (also moves ``data`` to _original_data)
        object.__setattr__(self, "data", self._Normalizer(self, self.data, self._data_err))

    # ===============================================================
    # Flags

    @cached_property
    def has_distances(self) -> bool:
        """Whether the data has distances or is on-sky."""
        data_onsky = self.data["coord"].spherical.distance.unit.physical_type == "dimensionless"
        origin_onsky = self.origin.spherical.distance.unit.physical_type == "dimensionless"
        onsky: bool = data_onsky and origin_onsky
        return not onsky

    @property
    def has_kinematics(self) -> bool:
        """Whether the data has kinematics."""
        has_vs = "s" in self.data["coord"].data.differentials

        # For now can't do RadialDifferential
        if has_vs:
            has_vs &= not isinstance(self.data["coord"].data.differentials["s"], RadialDifferential)

        return has_vs

    # ===============================================================
    # Directly from Data

    def get_data_coords(self, minPmemb: Quantity = Quantity(0, u.percent), include_outliers: bool = True) -> SkyCoord:
        coords = self.data["coord"]
        # filter data
        use = self.data["Pmemb"] >= minPmemb
        if not include_outliers:
            use &= self.data["order"] >= 0

        return coords[use]

    @property
    def data_coords(self) -> SkyCoord:
        """Get ``coord`` from data table."""
        return self.get_data_coords(minPmemb=Quantity(0, u.percent), include_outliers=False)

    @cached_property
    def data_frame(self) -> BaseCoordinateFrame:
        """The `astropy.coordinates.BaseCoordinateFrame` of the data."""
        reptype = self.data["coord"].representation_type
        if not self.has_distances:
            reptype = getattr(reptype, "_unit_representation", reptype)

        # Get the frame from the data
        frame: BaseCoordinateFrame = self.data["coord"].frame.replicate_without_data(representation_type=reptype)

        return frame

    # ===============================================================
    # System stuff (fit dependent)

    @property
    def system_frame(self) -> BaseCoordinateFrame | None:
        """Return a system-centric frame (or None).

        Determined from the argument ``frame`` at initialization.
        If None (default) and the method ``fit`` has been called,
        then a system frame has been found and cached.
        """
        frame: BaseCoordinateFrame | None
        if self._system_frame is not None:
            frame = self._system_frame
        else:
            frame = self.cache.get("system_frame")  # Can be `None`

        return frame

    @property
    def _best_frame(self) -> BaseCoordinateFrame:
        """:attr:`Stream.system_frame` unless its `None`, else :attr:`Stream.data_frame`."""
        frame = self.system_frame if self.system_frame is not None else self.data_frame
        return frame

    def get_coords(self, *, minPmemb: Quantity = Quantity(0, u.percent), include_outliers: bool = True) -> SkyCoord:
        """Data coordinates transformed to ``system_frame`` (if there is one).

        Parameters
        ----------
        minPmemb : Quantity['percent'], optional keyword-only
            Minimum member probability to be considered part of the stream, by default all the data.
        include_outliers : bool, optional
            Whether to include labelled outliers, by default `True`.

        Returns
        -------
        SkyCoord
        """
        data_coords = self.get_data_coords(minPmemb=minPmemb, include_outliers=include_outliers)
        frame = self._best_frame

        c = data_coords.transform_to(frame)
        c.representation_type = frame.representation_type
        c.differential_type = frame.differential_type

        return c

    @property
    def coords(self) -> SkyCoord:
        """Data coordinates transformed to `Stream.system_frame` (if there is one)."""
        return self.get_coords(minPmemb=Quantity(0, u.percent), include_outliers=True)

    @property
    def coords_ord(self) -> SkyCoord:
        """The (ordered) coordinates of the arm.

        Note that this can be shorter than ``coords`` since it outliers are
        removed when fitting the track. # TODO! or tack them on at the end?
        """
        use: ndarray = self.data["order"] >= 0
        order: ndarray = self.data["order"][use]
        return cast(SkyCoord, self.coords[order])

    # ===============================================================
    # Cleaning Data

    def label_outliers(self, outlier_method="scipyKDTreeLOF", **kwargs: Any) -> None:
        """Detect and label outliers, setting their ``order`` to -1."""
        # TODO! more complete, including velocity data
        data = structured_to_unstructured(self.data_coords.data._values)

        if isinstance(outlier_method, str):
            outlier_method = OUTLIER_DETECTOR_CLASSES[outlier_method]()
        elif not isinstance(outlier_method, OutlierDetectorBase):
            raise TypeError

        isoutlier = outlier_method.fit_predict(data, data, **kwargs)

        self.data["order"][isoutlier] = -1

    # ===============================================================
    # Fitting Frame
    # Generally want to do on the whole stream instead.

    def fit_frame(
        self,
        fitter: RotatedFrameFitter | None = None,
        rot0: Quantity[u.deg] | None = Quantity(0, u.deg),
        *,
        force: bool = False,
        minPmemb: Quantity[u.percent] = Quantity(80, u.percent),
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
            raise ValueError("a frame has already beem fit. use ``force`` to re-fit.")

        if fitter is None:  # make default fitter
            # LOCAL
            from trackstream.fit.rotated_frame import RotatedFrameFitter

            fitter = RotatedFrameFitter(  # type: ignore
                origin=self.origin,
                frame=self.data_frame,
                frame_representation_type=UnitSphericalRepresentation,
            )

        use = (self.data["Pmemb"] >= minPmemb) & (self.data["order"] >= 0)
        fitted = fitter.fit(cast(SkyCoord, self.data_coords[use]), rot0=rot0, **kwargs)

        # cache
        self._cache["frame_fitter"] = fitter
        self._cache["frame_fit"] = fitted
        self._cache["system_frame"] = fitted.frame

        return fitted.frame

    # ===============================================================
    # Fitting Track

    @property
    def track(self) -> StreamArmTrack:
        """Stream track.

        Raises
        ------
        ValueError
            If track is not fit.
        """
        track = self.cache["track"]
        if track is None:
            raise ValueError("need to fit track. See ``arm.fit_track(...)``.")
        return track

    def fit_track(
        self,
        *,
        fitter: bool | FitterStreamArmTrack = True,
        tune: bool = True,
        force: bool = False,
        minPmemb: Quantity[u.percent] = Quantity(80, u.percent),
        **kwargs: Any,
    ) -> StreamArmTrack:
        r"""Fit `~trackstream.fit.StreamArmTrack` to this stream.

        Parameters
        ----------
        fitter : bool or `~trackstream.fit.FitterStreamArmTrack`, optional
        keyword-only
            The fitter to use, by default `True`.
        tune : bool, optional
            Whether to train the SOM without writing to its current state
            (`True`, default). Keeping the current state is useful for re-using
            the SOM.
        force : bool, optional keyword-only
            Whether to force a track to be fit, even if one already has been for
            this stream. By default `False`.
        minPmemb : Quantity['percent'], optional keyword-only
            Minimum member probability to be considered part of the stream when
            fitting the track, by default 80%.

        Returns
        -------
        `~trackstream.fit.StreamArmTrack`

        Raises
        ------
        ValueError
            If a track has already been fit and ``force`` is not `True`.
        """
        # Check if already fit
        if not force and self._cache["track"] is not None:
            raise ValueError("already fit. use ``force`` to re-fit.")

        # LOCAL
        from trackstream.fit.fitter import FitterStreamArmTrack

        # Get fitter instance
        if not isinstance(fitter, FitterStreamArmTrack):
            use_cached = fitter is True
            fitter = self.cache["track_fitter"]

            if not use_cached or not isinstance(fitter, FitterStreamArmTrack):
                fitter = FitterStreamArmTrack.from_stream(self)

        # if tuning, the original fitter is unaffected. This is useful for
        # saving the fitter and using it many times.
        if tune:
            fitter = copy.deepcopy(fitter)

        # FIT
        track = fitter.fit(self, minPmemb=minPmemb, **kwargs)

        # Cache
        self._cache["track_fitter"] = fitter
        self._cache["track"] = track

        # Add ordering to data table
        order = fitter.cache["visit_order"]
        use = (self.data["Pmemb"] >= minPmemb) & (self.data["order"] >= 0)
        all_order = np.arange(len(self))
        all_order[use] = all_order[use][order]  # this skip some
        all_order[~use] = -1
        self.data["order"] = all_order

        return track

    # ===============================================================
    # Misc

    def __base_repr__(self, max_lines: int | None = None) -> list:
        rs = super().__base_repr__(max_lines=max_lines)

        # 5) data table
        datarep: str = self.data._base_repr_(html=False, max_width=None, max_lines=max_lines)
        table: str = "\n\t".join(datarep.split("\n")[1:])
        rs.append("  Data:\n\t" + table)

        return rs
