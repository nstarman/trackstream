"""Stream arm track fit result."""


from __future__ import annotations

from abc import abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING, Any, Generic
import weakref

from astropy.coordinates import PhysicsSphericalRepresentation, SphericalRepresentation, UnitSphericalRepresentation
import astropy.units as u
from astropy.utils.metadata import MetaAttribute, MetaData
from astropy.utils.misc import indent

from trackstream.stream.base import StreamLikeT

if TYPE_CHECKING:
    from collections.abc import Callable
    from dataclasses import InitVar

    from astropy.coordinates import BaseCoordinateFrame, SkyCoord
    from astropy.units import Quantity, percent
    from interpolated_coordinates import InterpolatedSkyCoord

    from trackstream.track.path import Path, path_moments

__all__: list[str] = []


##############################################################################
# CODE
##############################################################################


class StreamArmTrackBase(Generic[StreamLikeT]):
    """ABC of stream arm tracks."""

    @property
    @abstractmethod
    def stream(self) -> StreamLikeT:
        """The track's stream."""
        raise NotImplementedError


#####################################################################


class StreamArmTrack(StreamArmTrackBase[StreamLikeT]):
    """A stream track interpolation as function of arc length.

    Parameters
    ----------
    path : `~trackstream.utils.path.Path` instance
        Paths are an affine-parameterized position and distribution.
    name : str or None, optional keyword-only
        The name of the track.
    meta : dict[Any, Any]
        Metadata. Can include the meta-attributes
        ``visit_order``, ``som``, and ``kalman``.
    """

    meta = MetaData()

    # MetaAttributes (not type annotated b/c attrs treats as field)
    som = MetaAttribute()
    visit_order = MetaAttribute()
    kalman = MetaAttribute()

    # ===============================================================
    # Initialization

    stream_ref: InitVar[weakref.ReferenceType[StreamLikeT]]
    path: Path
    name: str | None

    def __init__(
        self,
        stream: StreamLikeT,
        path: Path,
        *,
        name: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "name", name)

        # set the MetaAttribute(s)
        self._meta: dict[str, Any]  # set by MetaData
        object.__setattr__(self, "_meta", {} if meta is None else meta)
        for attr in list(self._meta):
            descr = getattr(self.__class__, attr, None)
            if isinstance(descr, MetaAttribute):
                descr.__set__(self, self._meta.pop(attr))

        self.__post_init__(stream_ref=stream)

    def __post_init__(self, stream_ref: weakref.ReferenceType[StreamLikeT] | StreamLikeT) -> None:
        self._stream_ref: weakref.ReferenceType[StreamLikeT]
        sref = weakref.ref(stream_ref) if not isinstance(stream_ref, weakref.ReferenceType) else stream_ref
        object.__setattr__(self, "_stream_ref", sref)

    # ===============================================================

    @property
    def stream(self) -> StreamLikeT:
        """The `~trackstream.Stream`, or `None` if the weak reference is broken."""
        strm = self._stream_ref()
        if strm is None:
            msg = "the reference to the stream is broken"
            raise AttributeError(msg)
        return strm

    @property
    def origin(self) -> SkyCoord:
        """The origin of the track."""
        return self.stream.origin

    @property
    def full_name(self) -> str | None:
        """The full name of the track."""
        return self.name

    @property
    def coords(self) -> InterpolatedSkyCoord:
        """The path's central track."""
        return self.path.data

    @property
    def affine(self) -> Quantity:
        """The affine interpolation parameter."""
        return self.path.affine

    @property
    def frame(self) -> BaseCoordinateFrame:
        """The coordinate frame of the track."""
        return self.path.frame

    @cached_property
    def has_distances(self) -> bool:
        """Whether the data is in 3D or is on-sky."""
        out = not issubclass(self.coords.representation_type, UnitSphericalRepresentation)
        if issubclass(self.coords.representation_type, SphericalRepresentation | PhysicsSphericalRepresentation):
            out &= self.coords.distance.unit != u.one
        return out

    @cached_property
    def has_kinematics(self) -> bool:
        """Whether the track has kinematics."""
        return "s" in self.coords.data.differentials

    #######################################################
    # Math on the Track

    def __call__(self, affine: Quantity | None = None, *, angular: bool = False) -> path_moments:
        """Get discrete points along interpolated stream track.

        Parameters
        ----------
        affine : `~astropy.units.Quantity` array-like or None, optional
            The affine interpolation parameter. If None (default), return
            path moments evaluated at all "tick" interpolation points.
        angular : bool, optional keyword-only
            Whether to compute on-sky or real-space.

        Returns
        -------
        `trackstream.utils.path.path_moments`
            Realized from the ``.path`` attribute.
        """
        # TODO: add amplitude (density)
        return self.path(affine=affine, angular=angular)

    def probability(
        self,
        point: SkyCoord,
        background_model: Callable[[SkyCoord], Quantity[percent]] | None = None,
        *,
        angular: bool = False,
        affine: Quantity | None = None,
    ) -> Quantity[percent]:
        """Probability point is part of the stream."""
        raise NotImplementedError

    #######################################################
    # misc

    def __repr__(self) -> str:
        """Return string representation."""
        rs = []

        # 0) header (standard repr)
        header: str = object.__repr__(self)
        frame_name = self.frame.__class__.__name__
        rep_name = self.coords.representation_type.__name__
        header = header.replace("StreamArmTrack", f"StreamArmTrack ({frame_name}|{rep_name})")
        rs.append(header)

        # 1) name
        name = str(self.name)
        rs.append("  Name: " + name)

        # 2) data
        rs.append(indent(repr(self.path.data), width=2))

        return "\n".join(rs)

    def __len__(self) -> int:
        return len(self.path)
