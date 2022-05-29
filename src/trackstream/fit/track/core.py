# -*- coding: utf-8 -*-

"""Stream track fit result."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import weakref
from abc import abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING, Callable, Optional, Type

# THIRD PARTY
import astropy.units as u
from astropy.coordinates import (
    BaseCoordinateFrame,
    BaseDifferential,
    BaseRepresentation,
    SkyCoord,
    UnitSphericalRepresentation,
)
from astropy.units import Quantity
from astropy.utils.metadata import MetaAttribute, MetaData
from astropy.utils.misc import indent
from attrs import define, field
from interpolated_coordinates import InterpolatedSkyCoord

# LOCAL
from .visualization import StreamArmTrackPlotDescriptor
from trackstream.base import (
    FramedBase,
    frame_differential_type_factory,
    frame_representation_type_factory,
)
from trackstream.fit.path import Path, path_moments

if TYPE_CHECKING:
    # LOCAL
    from trackstream.stream.base import StreamBase
    from trackstream.stream.core import StreamArm


__all__ = ["StreamArmTrackBase", "StreamArmTrack"]


##############################################################################
# CODE
##############################################################################


@define(frozen=True, slots=False, init=False)
class StreamArmTrackBase:
    @property
    @abstractmethod
    def stream(self) -> "StreamBase":
        """The track's stream"""


#####################################################################


@define(frozen=True, slots=False, init=True)
class StreamArmTrack(StreamArmTrackBase, FramedBase):
    """A stream track interpolation as function of arc length.

    The track is `callable`, returning a |Frame|.

    Parameters
    ----------
    stream : `~trackstream.stream.Stream`
    path : `~trackstream.utils.path.Path`

    name : str or None, optional keyword-only
        The name of the track.
    **meta : Any
        Metadata. Can include the meta-attributes
        ``visit_order``, ``som``, and ``kalman``.
    """

    meta = MetaData()
    plot = StreamArmTrackPlotDescriptor()

    # MetaAttributes (not type annotated b/c attrs treats as field)
    som = MetaAttribute()
    visit_order = MetaAttribute()
    kalman = MetaAttribute()

    # ===============================================================

    _stream_ref: weakref.ReferenceType = field(
        converter=lambda x: weakref.ref(x)
    )  # turned into `stream`
    path: Path = field()
    name: Optional[str] = field(kw_only=True)
    _meta: dict = field(factory=dict, kw_only=True)

    frame: BaseCoordinateFrame = field(init=False)
    frame_representation_type: Type[BaseRepresentation] = field(
        init=False, default=frame_representation_type_factory
    )
    frame_differential_type: Optional[Type[BaseDifferential]] = field(
        init=False, default=frame_differential_type_factory
    )

    @frame.default
    def _frame_factory(self):
        return self.path.frame

    def __attrs_post_init__(self):
        super().__attrs_post_init__()

        # set the MetaAttribute(s)
        for attr in list(self._meta):
            descr = getattr(self.__class__, attr, None)
            if isinstance(descr, MetaAttribute):
                descr.__set__(self, self._meta.pop(attr))

    # ===============================================================

    @property
    def stream(self) -> "StreamArm":
        """The `~trackstream.Stream`, or `None` if the weak reference is broken."""
        strm = self._stream_ref()
        if strm is None:
            raise AttributeError("the reference to the stream is broken")
        return strm

    @property
    def origin(self):
        return self.stream.origin

    @property
    def full_name(self) -> Optional[str]:
        return self.name

    @property
    def coords(self) -> InterpolatedSkyCoord:
        """The path's central track."""
        return self.path.data

    @property
    def coords_ord(self) -> InterpolatedSkyCoord:
        """The path's central track."""
        return self.path.data

    @property
    def affine(self) -> Quantity:
        return self.path.affine

    @property
    def system_frame(self):
        return self.frame

    @cached_property
    def has_distances(self) -> bool:
        """Whether the data has distances or is on-sky"""
        # TODO a more robust check
        data_onsky = issubclass(type(self.coords.data), UnitSphericalRepresentation)
        return not data_onsky

    @cached_property
    def has_kinematics(self) -> bool:
        return "s" in self.coords.data.differentials

    #######################################################
    # Math on the Track

    def __call__(self, affine: Optional[Quantity] = None, *, angular: bool = False) -> path_moments:
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
        # TODO! add amplitude (density)
        return self.path(affine=affine, angular=angular)

    def probability(
        self,
        point: SkyCoord,
        background_model: Optional[Callable[[SkyCoord], Quantity[u.percent]]] = None,
        *,
        angular: bool = False,
        affine: Optional[Quantity] = None,
    ) -> Quantity[u.percent]:
        """Probability point is part of the stream.

        .. todo:: angular probability

        """
        # # Background probability
        # Pb = background_model(point) if background_model is not None else 0.0

        # angular = False  # TODO: angular probability
        # afn = self.path.closest_affine_to_point(point, angular=False, affine=affine)
        # pt_w = getattr(self.path, "width_angular" if angular else "width")(afn)
        # sep = getattr(self.path, "separation" if angular else "separation_3d")(
        #     point,
        #     interpolate=False,
        #     affine=afn,
        # )

        # # cov = 1  # Assumption
        # pdf = exp(-0.5 * sep ** 2) / power(2 * pi, 3.0 / 2)
        # # TODO! multidimensional PDF

        raise NotImplementedError

    #######################################################
    # misc

    def __repr__(self) -> str:
        """String representation."""
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
