"""Stream arm classes.

Stream arms are descriptors on a `trackstrea.Stream` class.

"""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, TypedDict, cast

# THIRD PARTY
import astropy.units as u
from astropy.coordinates import (
    BaseCoordinateFrame,
    SkyCoord,
    UnitSphericalRepresentation,
)
from astropy.coordinates import concatenate as concatenate_coords
from astropy.table import QTable
from astropy.units import Quantity
from attrs import define, field

# LOCAL
from .base import StreamBase
from .core import StreamArm
from .visualization import StreamPlotDescriptor
from trackstream._typing import CoordinateLikeType
from trackstream.base import CollectionBase
from trackstream.fit import FitterStreamArmTrack, StreamArmTrack
from trackstream.fit.track.plural import StreamArmsTracks, StreamTrack
from trackstream.utils._attrs import _cache_factory, convert_if_none

if TYPE_CHECKING:
    # LOCAL
    from trackstream.fit.path import path_moments
    from trackstream.fit.rotated_frame import FrameOptimizeResult, RotatedFrameFitter
    from trackstream.fit.track.plural import StreamArmsTrackBase


__all__ = ["StreamArmsBase", "StreamArms", "Stream"]


##############################################################################
# PARAMETERS


class _StreamCache(TypedDict):
    """Cache for Stream."""

    # arms' caches
    fitters: MappingProxyType | None
    # frame
    # system_frame: Optional[BaseCoordinateFrame]
    frame_fit: FrameOptimizeResult | None
    # frame_fitter: Optional[RotatedFrameFitter]
    # track
    track: StreamArmTrack | None


##############################################################################


@define(frozen=True, init=False, repr=False, slots=False)
class StreamArmsBase(CollectionBase[StreamArm]):
    """
    A collection of stream arms.
    For an object that brings together many stream arms, but acts like 1 stream, see ``Stream``.
    """


#####################################################################


@define(frozen=True, init=False, repr=False, slots=False)
class StreamArms(StreamArmsBase):
    """
    A collection of stream arms.
    For an object that brings together many stream arms, but acts like 1 stream, see ``Stream``.
    """


#####################################################################


@define(frozen=True, repr=False)
class Stream(StreamArmsBase, StreamBase):
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
    """

    plot = StreamPlotDescriptor()

    # ===============================================================
    # Initialization

    _data: dict = field(init=True, factory=dict, converter=dict)

    _cache: dict = field(
        kw_only=True,
        factory=_cache_factory(_StreamCache),
        converter=convert_if_none(_cache_factory(_StreamCache), deepcopy=True),
    )

    arms: StreamArms = field(init=False)

    @arms.default  # type: ignore
    def _arms_factory(self):
        return StreamArms(self._data)

    @_data.validator  # type: ignore
    def _data_validator(self, _, value: dict) -> None:
        if len(value) > 2:
            raise NotImplementedError(">2 stream arms are not yet supported")

    def __attrs_post_init__(self) -> None:
        # Make composite cache
        self._cache["fitters"] = MappingProxyType({k: arm.cache["track_fitter"] for k, arm in self.items()})

        # validate that all the data frames are the same
        data_frame = self.data_frame
        origin = self.origin
        system_frame = self.system_frame
        for name, arm in self.items():
            if not arm.data_frame == data_frame:
                raise ValueError(f"arm {name} data-frame must match {data_frame}")

            if not arm.origin == origin:
                raise ValueError(f"arm {name} origin must match {origin}")

            if not arm.system_frame == system_frame:
                raise ValueError(f"arm {name} origin must match {system_frame}")

        # For convenience, also attach an Arms of self

    @classmethod
    def from_data(
        cls,
        data: QTable,
        origin: SkyCoord,
        *,
        name: str | None = None,
        data_err: QTable | None = None,
        system_frame: CoordinateLikeType | None = None,
        caches: dict[str, dict | None] | None = None,
    ):
        # split data by arm
        data = data.group_by("tail")
        data.add_index("tail")

        # similarly for errors
        if data_err is not None:
            data_err = cast(QTable, data_err.group_by("tail"))
            data_err.add_index("tail")

        if caches is None:
            caches = {}

        # initialize each arm
        groups_keys = cast(QTable, data.groups.keys)
        arm_names: tuple[str, ...] = tuple(groups_keys["tail"])
        arms = {}
        for k in arm_names:
            arm = StreamArm(
                data.loc[k],
                origin=origin,
                name=k,
                data_err=None if data_err is None else data_err.loc[k],  # type: ignore
                system_frame=system_frame,  # type: ignore
                cache=caches.get(k),  # type: ignore
            )

            arms[k] = arm

        return cls(arms, name=name)

    # ===============================================================

    @property
    def cache(self) -> MappingProxyType:
        return MappingProxyType(self._cache)

    @property
    def data(self) -> MappingProxyType:
        return MappingProxyType(self._data)

    @property
    def data_frame(self) -> BaseCoordinateFrame:
        # all the arms must have the same frame
        key0 = tuple(self.keys())[0]
        data_frame = self[key0].data_frame
        return data_frame

    @property
    def origin(self) -> SkyCoord:
        # all the arms must have the same frame
        key0 = tuple(self.keys())[0]
        origin = self[key0].origin
        return origin

    @property
    def data_coords(self) -> SkyCoord:
        """The concatenated coordinates of all the arms."""
        if len(self.arms) > 1:
            sc = concatenate_coords([arm.data_coords for arm in self.values()])
        else:
            sc = next(iter(self.values())).data_coords
        return cast(SkyCoord, sc)

    @property
    def coords(self) -> SkyCoord:
        """The concatenated coordinates of all the arms."""
        if len(self.arms) > 1:
            sc = concatenate_coords([arm.coords for arm in self.values()])
        else:
            sc = next(iter(self.values())).coords
        return cast(SkyCoord, sc)

    @property
    def system_frame(self) -> BaseCoordinateFrame | None:
        """Return a system-centric frame (or None)."""
        key0 = tuple(self.keys())[0]
        frame = self[key0].system_frame
        return frame

    @property
    def has_distances(self) -> bool:
        return all(arm.has_distances for arm in self.values())

    @property
    def has_kinematics(self) -> bool:
        return all(arm.has_kinematics for arm in self.values())

    @property
    def coords_ord(self) -> SkyCoord:
        """The (ordered) coordinates of the arm."""
        arm0, arm1 = tuple(self.values())
        return concatenate_coords((arm0.coords_ord[::-1], arm1.coords_ord))

    # ===============================================================
    # Cleaning Data

    def label_outliers(self, outlier_method="scipyKDTreeLOF", **kwargs) -> None:
        """Detect and label outliers, setting their ``order`` to -1."""
        for arm in self.values():
            arm.label_outliers(outlier_method, **kwargs)

    # ===============================================================
    # Fitting

    def fit_frame(
        self,
        fitter: RotatedFrameFitter | None = None,
        rot0: Quantity | None = Quantity(0, u.deg),
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
        if any(f is not None for f in self._system_frame.values()):
            raise TypeError("a system frame was given at initialization.")
        elif self.system_frame is not None and not force:
            raise ValueError("a frame has already beem fit. use ``force`` to re-fit.")

        if fitter is None:
            # LOCAL
            from trackstream.fit.rotated_frame import RotatedFrameFitter

            fitter = RotatedFrameFitter(  # type: ignore
                origin=self.origin,
                frame=self.data_frame,
                frame_representation_type=UnitSphericalRepresentation,
            )

        fitted = fitter.fit(self.data_coords, rot0=rot0, **kwargs)

        # Cache
        self._cache["frame_fit"] = fitted
        for arm in self.values():
            arm._cache["frame_fitter"] = fitter
            arm._cache["frame_fit"] = fitted
            arm._cache["system_frame"] = fitted.frame

        return fitted.frame

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
            raise ValueError("need to fit track. See ``Stream.fit_track(...)``.")
        return track

    def fit_track(
        self,
        fitters: bool | Mapping[str, bool | FitterStreamArmTrack] = True,
        tune: bool = True,
        force: bool = False,
        composite: bool = True,
        **kwargs: Any,
    ) -> StreamArmsTrackBase:
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
            Passed to :meth:`trackstream.FitterStreamArmTrack.fit`.

        Returns
        -------
        `trackstream.StreamArmTrack`

        Raises
        ------
        ValueError
            If a frame has already been fit and ``force`` is not `True`.
        ValueError
            If ``onsky`` is False and the stream does not have distance
            information.
        """
        # Check if already fit
        if not force and self.cache["track"] is not None:
            raise ValueError("already fit. use ``force`` to re-fit.")

        if len(self.arms) > 2:
            raise NotImplementedError("TODO")

        # broadcast bool -> Dict[arm_name, bool]
        if not isinstance(fitters, Mapping):
            fitters = {k: fitters for k in self.keys()}

        # Fit all tracks
        tracks = {}
        for k, arm in self.items():
            tracks[k] = arm.fit_track(fitter=fitters.get(k, True), tune=tune, force=force, **kwargs)

        # -------------------
        # Currently only two arms are supported, so the tracks can be combined
        # together into a single path. One arm needs to be in reverse order to
        # be indexed toward origin, not away.

        if composite:
            track = StreamTrack.from_stream(self, name=(self.full_name or "").lstrip())
        else:
            track = StreamArmsTracks(tracks, name=(self.full_name or "").lstrip())

        self._cache["track"] = track
        self._cache["fitters"] = MappingProxyType({k: arm.cache["track_fitter"] for k, arm in self.items()})

        return track

    # ===============================================================
    # Math on the Track (requires fitting track)

    def predict_track(self, affine: u.Quantity | None = None, *, angular: bool = False) -> path_moments:
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

    def __len__(self) -> int:
        return sum(map(len, self.values()))

    def __base_repr__(self, max_lines: int | None = None) -> list:
        rs = super().__base_repr__(max_lines=max_lines)

        # 5) contained streams
        datarepr = (
            f"{name}:\n\t\t"
            + "\n\t\t".join(
                arm.data._base_repr_(html=False, max_width=None, max_lines=arm._data_max_lines).split("\n")[1:]
            )
            for name, arm in self.items()
        )
        rs.append("  Streams:\n\t" + "\n\t".join(datarepr))

        return rs

    def __repr__(self):
        return StreamBase.__repr__(self)
