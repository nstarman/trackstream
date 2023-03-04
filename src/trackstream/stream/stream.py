"""Stream arm classes.

Stream arms are descriptors on a `trackstrea.Stream` class.
"""


from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import asdict, dataclass, field, fields
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    NoReturn,
    TypedDict,
    TypeVar,
    cast,
    final,
)

from astropy.coordinates import BaseCoordinateFrame, SkyCoord
from astropy.coordinates import concatenate as concatenate_coords
from bound_class.core.descriptors import BoundDescriptor

from trackstream.stream.base import Flags, StreamBase
from trackstream.stream.core import StreamArm
from trackstream.stream.plural import StreamArms, StreamArmsBase
from trackstream.utils.coord_utils import get_frame, parse_framelike
from trackstream.utils.descriptors.cache import CacheProperty

if TYPE_CHECKING:
    from astropy.table import Column, QTable

    from trackstream._typing import FrameLikeType
    from trackstream.clean.base import OutlierDetectorBase
    from trackstream.common import CollectionBase
    from trackstream.frame.fit.result import FrameOptimizeResult
    from trackstream.track.core import StreamArmTrack
    from trackstream.track.fit import FitterStreamArmTrack
    from trackstream.track.plural import StreamArmsTrackBase


__all__ = ["Stream"]


##############################################################################
# PARAMETERS

Self = TypeVar("Self", bound="CollectionBase")


class _StreamCache(TypedDict):
    """Cache for Stream."""

    # frame
    frame_fit_result: FrameOptimizeResult | None
    # track
    track: StreamArmTrack | None
    fitters: MappingProxyType[str, bool | FitterStreamArmTrack] | None  # arms' caches


##############################################################################


class StreamArmsDescriptor(StreamArms, BoundDescriptor["Stream"]):
    """Stream arms descriptor."""

    def __init__(self, store_in: Literal["__dict__", "_attrs_"] | None = "__dict__") -> None:
        object.__setattr__(self, "store_in", store_in)

    @property
    def _data(self) -> dict[str, StreamArm]:
        return self.enclosing._data  # noqa: SLF001

    @property
    def name(self) -> str | None:
        """The name of the attribute."""
        return self.enclosing.name

    def __set__(self, obj: object, value: Any) -> NoReturn:
        raise AttributeError


@dataclass(frozen=True)
class StreamFlags(Flags, BoundDescriptor["Stream"]):
    """Flags for a `Stream`."""

    store_in: Literal["__dict__"] = field(default="__dict__", repr=False)

    def set(self, **kwargs: Any) -> None:  # noqa: A003
        """Set flags.

        Parameters
        ----------
        kwargs
            Flags to set.
        """
        super().set(**kwargs)

        for v in self.__self__.values():
            v.flags.set(**kwargs)

    def asdict(self) -> dict[str, Any]:
        """Return flags as a dictionary."""
        fns = tuple(f.name for f in fields(Flags))
        return {k: v for k, v in asdict(self).items() if k in fns}


# ===================================================================


@final
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

    _CACHE_CLS: ClassVar[type] = _StreamCache
    cache = CacheProperty["StreamBase"]()
    arms = StreamArmsDescriptor()
    flags = StreamFlags()

    def __init__(
        self,
        data: dict[str, StreamArm] | None = None,
        /,
        *,
        name: str | None = None,
        **kwargs: StreamArm,
    ) -> None:
        self.name: str | None
        super().__init__(data, name=name, **kwargs)

        cache = CacheProperty._init_cache(self)  # noqa: SLF001
        self._cache: dict[str, Any]
        object.__setattr__(self, "_cache", cache)

        # validate data length
        if len(self._data) > 2:
            msg = ">2 stream arms are not yet supported"
            raise NotImplementedError(msg)

    @classmethod
    def from_data(
        cls: type[Self],
        data: QTable,
        origin: SkyCoord,
        *,
        name: str | None = None,
        data_err: QTable | None = None,
        frame: FrameLikeType | None = None,
        caches: dict[str, dict[str, Any] | None] | None = None,
    ) -> Self:
        """Create a `Stream` from data.

        Parameters
        ----------
        data : `~astropy.table.QTable`
            The stream data.
        origin : `~astropy.coordinates.SkyCoord`
            The origin point of the stream (and rotated reference frame).
        name : str or None, optional keyword-only
            The name of the stream.
        data_err : `~astropy.table.QTable` (optional)
            Optionally separate table for the data errors.
        frame : `~astropy.coordinates.BaseCoordinateFrame` or None, optional keyword-only
            The stream frame. Locally linearizes the data.
            If `None` (default), need to fit for the frame.
        caches : dict[str, dict[str, Any] | None] or None, optional keyword-only
            The cache data.

        Returns
        -------
        Stream
        """
        # split data by arm
        data = data.group_by("arm")
        data.add_index("arm")

        # similarly for errors
        if data_err is not None:
            data_err = cast("QTable", data_err.group_by("arm"))
            data_err.add_index("arm")

        if caches is None:
            caches = {}

        # resolve frame
        if frame is not None:
            frame = parse_framelike(frame)

        # initialize each arm
        groups_keys = cast("Column", data.groups.keys)
        arm_names: tuple[str, ...] = tuple(groups_keys["arm"])
        arms: dict[str, StreamArm] = {}
        for k in arm_names:
            arm = StreamArm.from_format(
                data.loc[k],
                origin=origin,
                name=k,
                data_err=None if data_err is None else data_err.loc[k],
                frame=frame,
                _cache=caches.get(k),
                format="astropy.table",
            )
            arms[k] = arm

        return cls(arms, name=name)

    # ===============================================================

    @property
    def data_frame(self) -> BaseCoordinateFrame:
        """The data frame."""
        # all the arms must have the same frame
        return self[self._k0].data_frame

    @property
    def origin(self) -> SkyCoord:
        """The origin of the stream."""
        # all the arms must have the same frame
        return self[self._k0].origin

    @property
    def data_coords(self) -> SkyCoord:
        """The concatenated coordinates of all the arms."""
        return (
            concatenate_coords([arm.data_coords for arm in self.values()])
            if len(self._data) > 1
            else self._v0.data_coords
        )

    @property
    def coords(self) -> SkyCoord:
        """The concatenated coordinates of all the arms."""
        if len(self._data) == 1:
            sc = self._v0.coords
        else:
            arm0, arm1 = tuple(self.values())
            sc = concatenate_coords((arm0.coords[::-1], arm1.coords))
        return sc  # noqa: RET504

    @property
    def frame(self) -> BaseCoordinateFrame | None:
        """Return a system-centric frame (or None)."""
        return self[self._k0].frame

    @property
    def has_distances(self) -> bool:
        """Return `True` if all the arms have distances."""
        return all(arm.has_distances for arm in self.values())

    @property
    def has_kinematics(self) -> bool:
        """Return `True` if all the arms have kinematics."""
        return all(arm.has_kinematics for arm in self.values())

    # ===============================================================
    # Cleaning Data

    def mask_outliers(self, outlier_method: str | OutlierDetectorBase = "ScipyKDTreeLOF", **kwargs: Any) -> None:
        """Detect and label outliers, setting their ``order`` to -1."""
        for arm in self.values():
            arm.mask_outliers(outlier_method, **kwargs)

    # ===============================================================
    # Track

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
            msg = "need to fit track. See ``Stream.fit_track(...)``."
            raise ValueError(msg)
        return track

    def fit_track(
        self,
        fitters: bool | Mapping[str, bool | FitterStreamArmTrack] = True,  # noqa: FBT002
        *,
        tune: bool = True,
        force: bool = False,
        composite: bool = True,
        **kwargs: Any,
    ) -> StreamArmsTrackBase:
        """Make a stream track.

        Parameters
        ----------
        fitters : bool or dict[str, bool or ``FitterStreamArmTrack``]
            If `True`, make fitters.
            if `dict` use the given fitters.

        tune : bool, optional keyword-only
            If `True` (default), tune the fitters, don't retrain them.
        force : bool, optional keyword-only
            If `True`, re-fit the track.
        composite : bool, optional keyword-only
            If `True` (default), make a composite track.
        **kwargs : Any optional keyword-only
            Passed to the fitters.

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
            msg = "already fit. use ``force`` to re-fit."
            raise ValueError(msg)

        if len(self._data) > 2:
            raise NotImplementedError("TODO")  # noqa: EM101

        # Broadcast bool -> Dict[arm_name, bool]
        if not isinstance(fitters, Mapping):
            fitters = {k: fitters for k in self}

        # Fit all tracks
        tracks = {}
        for k, arm in self.items():
            kw = copy.deepcopy(kwargs)  # ensure kwargs are decoupled
            tracks[k] = arm.fit_track(fitter=fitters.get(k, True), tune=tune, force=force, **kw)

        # -------------------
        # Currently only two arms are supported, so the tracks can be combined
        # together into a single path. One arm needs to be in reverse order to
        # be indexed toward origin, not away.

        # LOCAL
        from trackstream.track.plural import StreamArmsTracks, StreamTrack

        track = (
            StreamTrack.from_stream(self, name=(self.full_name or "").lstrip())
            if composite
            else StreamArmsTracks(tracks, name=(self.full_name or "").lstrip())
        )

        self._cache["track"] = track
        self._cache["fitters"] = MappingProxyType({k: arm.cache["track_fitter"] for k, arm in self.items()})

        return track

    # ===============================================================
    # Misc

    def __len__(self) -> int:
        return sum(map(len, self.values()))

    def _base_repr_(self, max_lines: int | None = None) -> list[str]:
        rs = super()._base_repr_(max_lines=max_lines)

        # 5) contained streams
        datarepr = (
            f"{name}:\n\t\t"
            + "\n\t\t".join(
                arm.data._base_repr_(html=False, max_width=None, max_lines=10).split("\n")[1:],  # noqa: SLF001
            )
            for name, arm in self.items()
        )
        rs.append("  Streams:\n\t" + "\n\t".join(datarepr))

        return rs

    def __repr__(self) -> str:
        return StreamBase.__repr__(self)


@get_frame.register
def _get_frame_stream(stream: Stream, /) -> BaseCoordinateFrame:
    if stream.frame is None:
        # LOCAL
        from trackstream.stream.base import FRAME_NONE_ERR

        raise FRAME_NONE_ERR

    return stream.frame
