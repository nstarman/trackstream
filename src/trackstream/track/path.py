"""Path are an affine-parameterized path."""

from __future__ import annotations

from dataclasses import InitVar, dataclass
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any, ClassVar, Literal, NamedTuple, cast, overload

import astropy.coordinates as coords
import astropy.units as u
from astropy.utils.decorators import format_doc
from astropy.utils.metadata import MetaData
import interpolated_coordinates as icoords
import numpy as np
from overload_numpy import NPArrayOverloadMixin, NumPyOverloader
from scipy.optimize import OptimizeResult, minimize_scalar

from trackstream.track.width.interpolated import InterpolatedWidths
from trackstream.utils.coord_utils import get_frame, parse_framelike

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from interpolated_coordinates.utils import InterpolatedUnivariateSplinewithUnits as IUSU  # noqa: N817

    from trackstream._typing import CoordinateType, FrameLikeType
    from trackstream.track.width.core import BaseWidth
    from trackstream.track.width.plural import Widths

__all__ = ["Path", "path_moments"]


##############################################################################
# PARAMETERS

PATH_FUNCS = NumPyOverloader()


class path_moments(NamedTuple):
    """Moments of a path."""

    mean: coords.SkyCoord
    width: u.Quantity | Widths | None


##############################################################################
# CODE
##############################################################################


@dataclass(frozen=True)
class Path(NPArrayOverloadMixin):
    """Paths are an affine-parameterized position and distribution.

    Parameters
    ----------
    path : coord-like array or `BaseRepresentation` subclass instance
        The central path of the path.

        If `BaseRepresentation`, then need to pass `frame`.
        If not interpolated (`InterpolatedSkyCoord`,
        `InterpolatedCoordinateFrame` or `InterpolatedRepresentation`), then
        need to pass `affine`.
        The best is `InterpolatedSkyCoord` or `InterpolatedCoordinateFrame`

    width : `Quantity` scalar or array or callable
        The width around `path`.

        Must have dimensions of 'length' or 'angle'.
        If array, must match `path` length
        If callable, must accept `path` affine as 1st (and only mandatory)
        argument

    amplitude : `Quantity` scalar or array or callable
        The density.

    name : str (optional, keyword-only)
        Name of the path

    Other Parameters
    ----------------
    affine : `Quantity` array (optional, keyword-only)
        Affine parameter along `path`.
        Only used if `path` is not already interpolated.

    frame : frame-like or None (optional, keyword-only)
        The preferred frame of the data (`path`)
        unless `path` has a frame (is not `BaseRepresentation`).
        If `path` is `BaseRepresentation`, then it is assumed in this frame.

    Raises
    ------
    Exception
        if `path` is not already interpolated and affine is None

    """

    NP_OVERLOADS: ClassVar[NumPyOverloader] = PATH_FUNCS

    meta = MetaData(copy=True)

    data: icoords.InterpolatedSkyCoord
    width: InterpolatedWidths | None = None
    amplitude: Callable[[u.Quantity | None], u.Quantity] | None = None
    name: str | None = None
    metadata: InitVar[dict[str, Any] | None] = None

    def __post_init__(self, metadata: dict[str, Any] | None) -> None:
        # Validation
        if not isinstance(self.data, icoords.InterpolatedSkyCoord):
            msg = "use `Path.from_format` instead"
            raise TypeError(msg)
        if self.width is not None and not np.array_equal(self.data.affine, self.width.affine):
            msg = "data.affine != width.affine"
            raise ValueError(msg)

        # Init
        self._meta: dict[Any, Any]
        object.__setattr__(self, "_meta", metadata if metadata is not None else {})

        self.frame: coords.BaseCoordinateFrame
        object.__setattr__(self, "frame", get_frame(self.data))

    # ===============================================================

    @property
    def affine(self) -> u.Quantity:
        """Affine parameter along ``path``."""
        return self.data.affine

    # ===============================================================
    # I/O

    @singledispatchmethod
    @classmethod
    def from_format(  # noqa: PLR0913
        cls,
        data: object,  # noqa: ARG003
        width: u.Quantity | Callable | None,  # noqa: ARG003
        amplitude: Callable | None = None,  # noqa: ARG003
        *,
        affine: u.Quantity,  # noqa: ARG003
        name: str | None = None,  # noqa: ARG003
        metadata: dict[str, Any] | None = None,  # noqa: ARG003
        **kwargs: Any,  # noqa: ARG003
    ) -> Path:  # https://github.com/python/mypy/issues/11727
        """Create a path from an object.

        Parameters
        ----------
        data : object
            The data to create the Path from.
        width : `Quantity` scalar or array or callable, optional
            The width around the Path.
        amplitude : callable, optional
            The density.
        affine : `Quantity` array, keyword-only
            Affine parameter along the Path.
        name : str, optional keyword-only
            Name of the Path.
        metadata : dict, optional keyword-only
            Metadata of the Path.
        **kwargs : Any
            Additional keyword arguments to pass to the constructor.

        Returns
        -------
        path : `Path`

        """
        msg = "not dispatched"
        raise NotImplementedError(msg)

    @from_format.register(coords.SkyCoord)
    @classmethod
    def _from_format_skycoord(  # noqa: PLR0913
        cls,
        data: coords.SkyCoord,
        width: Widths | None,
        amplitude: Callable | None = None,
        *,
        affine: u.Quantity,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
        **_: Any,
    ) -> Path:
        if isinstance(data, icoords.InterpolatedSkyCoord) and data.affine != affine:
            raise ValueError
        isc = icoords.InterpolatedSkyCoord(data, affine=affine)

        # munge width to right format
        ws = InterpolatedWidths.from_format(width, affine=affine)

        return cls(isc, width=ws, amplitude=amplitude, name=name, metadata=metadata or {})

    @from_format.register(coords.BaseCoordinateFrame)
    @from_format.register(icoords.InterpolatedCoordinateFrame)
    @classmethod
    def _from_format_frame(  # noqa: PLR0913
        cls,
        data: coords.BaseCoordinateFrame | icoords.InterpolatedCoordinateFrame,
        width: Widths | None,
        amplitude: Callable | None = None,
        *,
        affine: u.Quantity,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
        **_: Any,
    ) -> Path:
        if isinstance(data, icoords.InterpolatedCoordinateFrame) and not np.array_equal(data.affine, affine):
            raise ValueError

        data_if = icoords.InterpolatedCoordinateFrame(data, affine=affine)
        data_sc = coords.SkyCoord(data_if, copy=False)

        return cls.from_format(data_sc, width=width, amplitude=amplitude, affine=affine, name=name, metadata=metadata)

    @from_format.register(coords.BaseRepresentation)
    @from_format.register(icoords.InterpolatedRepresentation)
    @classmethod
    def _from_format_representation(  # noqa: PLR0913
        cls,
        data: coords.BaseRepresentation | icoords.InterpolatedRepresentation,
        width: Widths | None,
        amplitude: Callable | None = None,
        *,
        affine: u.Quantity,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
        frame: FrameLikeType,
        **_: Any,
    ) -> Path:
        theframe = parse_framelike(frame)
        rep_type = data.__class__
        dif_type = data.differentials["s"].__class__ if "s" in data.differentials else None

        data_f = theframe.realize_frame(data, representation_type=rep_type, differential_type=dif_type)
        return cls.from_format(data_f, width=width, amplitude=amplitude, affine=affine, name=name, metadata=metadata)

    # ===============================================================
    # Magic Methods

    def __len__(self) -> int:
        return len(self.data)

    # ===============================================================
    # Math on the Track!

    def __call__(self, affine: u.Quantity | None = None, *, angular: bool = False) -> path_moments:
        """Call.

        Parameters
        ----------
        affine : `~astropy.units.Quantity` array-like or None, optional
            The affine interpolation parameter. If None (default), return
            all positions.
        angular : bool, optional keyword-only
            Whether to return the width in units of length or the on-sky width
            in angular units.

        Returns
        -------
        `.path_moments`

        See Also
        --------
        trackstream.utils.path.Path.position
            For the 1st element of the `trackstream.utils.path.path_moments`.
        trackstream.utils.path.Path.width
            For the 2nd element of the `trackstream.utils.path.path_moments`.

        """
        mean = self.position(affine)
        width = (self.width(affine) if not angular else self.width_angular(affine)) if self.width is not None else None
        # TODO: add amplitude (density)
        return path_moments(mean, width)

    # -----------------------

    def position(
        self,
        affine: u.Quantity | None = None,
    ) -> icoords.InterpolatedSkyCoord | coords.SkyCoord:
        """Return the position on the track.

        Parameters
        ----------
        affine : `~astropy.units.Quantity` array-like or None, optional
            The affine interpolation parameter. If None (default), return
            all positions.

        Returns
        -------
        `~trackstream.utils.interpolated_coordinates.InterpolatedSkyCoord`
            Path position at ``affine``.

        See Also
        --------
        trackstream.utils.path.Path.data
            This is the same as ``.data(affine)``.

        """
        return self.data(affine)

    # -----------------------

    def width_angular(self, affine: u.Quantity | None = None) -> u.Quantity:
        """Return the (1-sigma) angular width of the track at affine points.

        Parameters
        ----------
        affine : `~astropy.units.Quantity` array-like or None, optional
            The affine interpolation parameter. If None (default), return
            angular width evaluated at all "tick" interpolation points.

        Returns
        -------
        `~astropy.unitss.Quantity`
            Path angular width evaluated at ``affine``.

        """
        if self.width is None:
            raise ValueError

        width: u.Quantity = self.width(affine)
        if width.unit.physical_type == "angle":
            return width

        # TODO: is there a more succinct Astropy func for this?
        r = self.data(affine).represent_as(coords.SphericalRepresentation)
        distance = r.distance.to_value(width.unit)

        return np.abs(np.arctan2(width.value, distance)) << u.rad

    # -----------------------------------------------------

    @overload
    def separation(
        self,
        point: CoordinateType,
        *,
        interpolate: Literal[True] = True,
        affine: u.Quantity | None = None,
    ) -> IUSU: ...

    @overload
    def separation(
        self,
        point: CoordinateType,
        *,
        interpolate: Literal[False] = False,
        affine: u.Quantity | None = None,
    ) -> coords.Angle: ...

    def separation(
        self,
        point: CoordinateType,
        *,
        interpolate: bool = True,
        affine: u.Quantity | None = None,
    ) -> coords.Angle | IUSU:
        """Return the separation between the track and a point.

        Parameters
        ----------
        point : |SkyCoord| or |BaseCoordinateFrame|
            The point to calculate the separation to.
        interpolate : bool, optional keyword-only
            Whether to interpolate the separation. If True (default), return
            an `~scipy.interpolate.InterpolatedUnivariateSpline` object.
        affine : `~astropy.units.Quantity` array-like or None, optional
            The affine interpolation parameter.

        """
        return self.data.separation(point, interpolate=interpolate, affine=affine)

    # -------------------------------------------

    @overload
    def separation_3d(
        self,
        point: CoordinateType,
        *,
        interpolate: Literal[True] = True,
        affine: u.Quantity | None = None,
    ) -> IUSU: ...

    @overload
    def separation_3d(
        self,
        point: CoordinateType,
        *,
        interpolate: Literal[False] = False,
        affine: u.Quantity | None = None,
    ) -> coords.Distance: ...

    @format_doc(icoords.InterpolatedSkyCoord.separation_3d.__doc__)
    def separation_3d(
        self,
        point: CoordinateType,
        *,
        interpolate: bool = True,
        affine: u.Quantity | None = None,
    ) -> coords.Distance | IUSU:
        """Return the 3D separation between the track and a point."""
        return self.data.separation_3d(point, interpolate=interpolate, affine=affine)

    # -----------------------------------------------------

    def _closest_res_to_point(
        self,
        point: CoordinateType,
        *,
        angular: bool = False,
        affine: u.Quantity | None = None,
    ) -> OptimizeResult:
        """Closest to stream, ignoring width."""
        sep_fn = (
            self.separation(point, interpolate=True, affine=affine)
            if angular
            else self.separation_3d(point, interpolate=True, affine=affine)
        )

        afn = self.affine if affine is None else affine
        res: OptimizeResult = minimize_scalar(
            lambda afn: sep_fn(afn).value,
            bounds=[afn.value.min(), afn.value.max()],
        )
        return res

    def closest_affine_to_point(
        self,
        point: CoordinateType,
        *,
        angular: bool = False,
        affine: u.Quantity | None = None,
    ) -> u.Quantity:
        """Closest affine, ignoring width."""
        afn = self.affine if affine is None else affine
        res = self._closest_res_to_point(point, angular=angular, affine=afn)
        return res.x << afn.unit

    def closest_position_to_point(
        self,
        point: CoordinateType,
        *,
        angular: bool = False,
        affine: u.Quantity | None = None,
    ) -> coords.SkyCoord:
        """Closest point, ignoring width."""
        return self.position(self.closest_affine_to_point(point, angular=angular, affine=affine))


##############################################################################
# NumPy Overloading


def _validate_concatenate_args(seqpaths: Sequence[Path], out: Path | None, dtype: np.dtype | None) -> None:
    N = len(seqpaths)
    if N == 0:
        msg = "need at least one array to concatenate"
        raise ValueError(msg)
    if N > 2:
        msg = "cannot concatenate more than 2 paths"
        raise ValueError(msg)

    if out is not None:
        msg = "out must be None"
        raise ValueError(msg)

    if dtype is not None:
        msg = "dtype must be None"
        raise ValueError(msg)


@Path.NP_OVERLOADS.implements(np.concatenate, dispatch_on=Path)
def concatenate(
    seqpaths: Sequence[Path],
    axis: int = 0,
    out: Path | None = None,
    dtype: np.dtype | None = None,
    _: str = "same_kind",
) -> Path:
    """Concatenate a sequence of Paths."""
    _validate_concatenate_args(seqpaths, out, dtype)
    if len(seqpaths) == 1:
        if axis != 0:
            msg = "axis must be 0 for 1 path"
            raise ValueError(msg)
        return seqpaths[0]

    # ----------------------------------------

    npth, ppth = seqpaths
    if npth.frame != ppth.frame:
        msg = "the paths must have the same frame"
        raise ValueError(msg)
    if npth.frame.representation_type != ppth.frame.representation_type:
        msg = "the paths must have the same representation_type"
        raise ValueError(msg)

    affine = cast("u.Quantity", np.concatenate((-npth.affine[::-1], ppth.affine)))

    # get representations, uninterpolated
    # TODO: add concatenated to InterpolatedSkyCoord
    nr = npth.data.data.data  # Representation
    if "s" in nr.differentials:
        nr.differentials["s"] = nr.differentials["s"].data
    pr = ppth.data.data.data
    if "s" in pr.differentials:
        pr.differentials["s"] = pr.differentials["s"].data
    r = coords.concatenate_representations((nr[::-1], pr))
    c = icoords.InterpolatedSkyCoord(npth.data.realize_frame(r, affine=affine))

    # Width
    negow = npth.width
    posow = ppth.width
    if negow is None and posow is None:
        width = None
    elif negow is None or posow is None:
        msg = "negow and posow is None"
        raise ValueError(msg)
    else:
        uninterp_width = np.concatenate((negow.uninterpolated[::-1], posow.uninterpolated))
        uninterp_width = cast("BaseWidth", uninterp_width)
        width = InterpolatedWidths.from_format(uninterp_width, affine=affine)

    # Name
    name = ppth.name if (npth.name == ppth.name) else f"{npth.name} | {ppth.name}"

    # Metadata
    # TODO: a better merge
    metadata = {"npth": npth.meta, "ppth": ppth.meta}

    # TODO: amplitude
    return Path(c, width=width, amplitude=None, name=name, metadata=metadata)


# ============================================================================


@get_frame.register
def _get_frame_path(path: Path) -> coords.BaseCoordinateFrame:
    return path.frame
