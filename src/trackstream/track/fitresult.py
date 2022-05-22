# -*- coding: utf-8 -*-

"""Stream track fit result."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import weakref
from functools import cached_property
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

# THIRD PARTY
import astropy.units as u
from astropy.coordinates import (
    BaseCoordinateFrame,
    SkyCoord,
    UnitSphericalRepresentation,
)
from astropy.units import Quantity
from astropy.utils.metadata import MetaAttribute, MetaData
from astropy.utils.misc import indent
from interpolated_coordinates import InterpolatedSkyCoord
from numpy import ndarray

# LOCAL
from .kalman import FirstOrderNewtonianKalmanFilter
from .path import Path, path_moments
from .som import SelfOrganizingMap1DBase
from .visualization import StreamTrackPlotDescriptor
from trackstream.base import CommonBase
from trackstream.utils.descriptors import TypedMetaAttribute

if TYPE_CHECKING:
    # LOCAL
    from trackstream.stream import Stream  # noqa: E402


__all__ = ["StreamTrack"]


##############################################################################
# CODE
##############################################################################


class StreamTrack(CommonBase):
    """A stream track interpolation as function of arc length.

    The track is Callable, returning a Frame.

    Parameters
    ----------
    stream : `~trackstream.stream.Stream` path : `~trackstream.utils.path.Path`
    origin : SkyCoord
        of the coordinate system (often the progenitor)

    name : str or None, optional keyword-only **meta : Any
        Metadata. Can include the meta-attributes
        ``visit_order``, ``som``, and ``kalman``.
    """

    _name: Optional[str]
    _meta: Dict[str, Any]
    meta = MetaData()

    visit_order: TypedMetaAttribute = TypedMetaAttribute[ndarray]()
    som: TypedMetaAttribute = TypedMetaAttribute[Dict[str, SelfOrganizingMap1DBase]]()
    kalman: TypedMetaAttribute = TypedMetaAttribute[Dict[str, FirstOrderNewtonianKalmanFilter]]()

    plot = StreamTrackPlotDescriptor()

    def __init__(
        self,
        stream: "Stream",
        path: Path,
        origin: SkyCoord,
        *,
        name: Optional[str] = None,
        **meta: Any,
    ) -> None:
        super().__init__(frame=path.frame, representation_type=None, differential_type=None)
        self._name = name
        self._stream_ref = weakref.ref(stream)  # reference to original stream

        # validation of types
        if not isinstance(path, Path):
            raise TypeError("`path` must be <Path>.")
        elif not isinstance(origin, (SkyCoord, BaseCoordinateFrame)):
            raise TypeError("`origin` must be <|SkyCoord|, |Frame|>.")

        # assign
        self._path: Path = path
        self._origin = origin

        # set the MetaAttribute(s)
        for attr in list(meta):
            descr = getattr(self.__class__, attr, None)
            if isinstance(descr, MetaAttribute):
                setattr(self, attr, meta.pop(attr))
        # and the meta
        self._meta.update(meta)

    @property
    def stream(self) -> Optional["Stream"]:
        """The `~trackstream.Stream`, or `None` if the weak reference is broken."""
        return self._stream_ref()

    @property
    def name(self) -> Optional[str]:
        """Return the stream-track name."""
        return self._name

    @property
    def full_name(self) -> Optional[str]:
        return self._name

    @property
    def path(self) -> Path:
        return self._path

    @property
    def coords(self) -> InterpolatedSkyCoord:
        """The path's central track."""
        return self._path.data

    @property
    def coords_ord(self) -> InterpolatedSkyCoord:
        """The path's central track."""
        return self._path.data

    @property
    def affine(self) -> Quantity:
        return self._path.affine

    @property
    def origin(self) -> SkyCoord:
        return self._origin

    @property
    def frame(self) -> BaseCoordinateFrame:
        crds = self.coords
        frame = crds.frame.replicate_without_data()
        frame.representation_type = crds.representation_type
        frame.differential_type = crds.differential_type
        return frame

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
        # afn = self._path.closest_affine_to_point(point, angular=False, affine=affine)
        # pt_w = getattr(self._path, "width_angular" if angular else "width")(afn)
        # sep = getattr(self._path, "separation" if angular else "separation_3d")(
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
        header = header.replace("StreamTrack", f"StreamTrack ({frame_name}|{rep_name})")
        rs.append(header)

        # 1) name
        name = str(self.name)
        rs.append("  Name: " + name)

        # 2) data
        rs.append(indent(repr(self._path.data), width=2))

        return "\n".join(rs)

    def __len__(self) -> int:
        return len(self._path)
