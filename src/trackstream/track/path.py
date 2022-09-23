"""Path are an affine-parameterized path."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from dataclasses import InitVar, dataclass
from functools import singledispatchmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    NamedTuple,
    Sequence,
    cast,
    overload,
)

# THIRD PARTY
import astropy.coordinates as coords
import astropy.units as u
import interpolated_coordinates as icoords
import numpy as np
from astropy.utils.decorators import format_doc
from astropy.utils.metadata import MetaData
from scipy.optimize import OptimizeResult, minimize_scalar

# LOCAL
from trackstream.track.width.core import BaseWidth
from trackstream.track.width.interpolated import InterpolatedWidths
from trackstream.utils.coord_utils import get_frame, parse_framelike
from trackstream.utils.numpy_overload import NumPyOverloader
from trackstream.utils.visualization import PlotDescriptorBase

if TYPE_CHECKING:
    # THIRD PARTY
    from interpolated_coordinates.utils import (
        InterpolatedUnivariateSplinewithUnits as IUSU,
    )

    # LOCAL
    from trackstream._typing import CoordinateType, FrameLikeType
    from trackstream.track.width.plural import Widths

__all__ = ["Path", "path_moments"]


##############################################################################
# PARAMETERS


class path_moments(NamedTuple):
    mean: coords.SkyCoord
    width: u.Quantity | Widths | None


##############################################################################
# CODE
##############################################################################


@dataclass(frozen=True)
class Path:
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

    meta = MetaData(copy=True)
    plot = PlotDescriptorBase["Path"]()

    data: icoords.InterpolatedSkyCoord
    width: InterpolatedWidths | None = None
    amplitude: Callable[[u.Quantity | None], u.Quantity] | None = None
    name: str | None = None
    metadata: InitVar[dict[str, Any] | None] = None

    def __post_init__(self, metadata: dict[str, Any] | None) -> None:
        # Validation
        if not isinstance(self.data, icoords.InterpolatedSkyCoord):
            raise TypeError("use `Path.from_format` instead")
        elif self.width is not None and not np.array_equal(self.data.affine, self.width.affine):
            raise ValueError("data.affine != width.affine")

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
    def from_format(
        cls,
        data: object,
        width: u.Quantity | Callable | None,
        amplitude: Callable | None = None,
        *,
        affine: u.Quantity,
        name: str | None = None,
        metadata: dict | None = None,
        **kwargs: Any,
    ) -> Any:  # https://github.com/python/mypy/issues/11727
        raise NotImplementedError("not dispatched")

    @from_format.register(coords.SkyCoord)
    @classmethod
    def _from_format_skycoord(
        cls,
        data: coords.SkyCoord,
        width: Widths | None,
        amplitude: Callable | None = None,
        *,
        affine: u.Quantity,
        name: str | None = None,
        metadata: dict | None = None,
        **kwargs: Any,
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
    def _from_format_frame(
        cls,
        data: coords.BaseCoordinateFrame | icoords.InterpolatedCoordinateFrame,
        width: Widths | None,
        amplitude: Callable | None = None,
        *,
        affine: u.Quantity,
        name: str | None = None,
        metadata: dict | None = None,
        **kwargs: Any,
    ) -> Path:
        if isinstance(data, icoords.InterpolatedCoordinateFrame) and not np.array_equal(data.affine, affine):
            raise ValueError

        data_if = icoords.InterpolatedCoordinateFrame(data, affine=affine)  # type: ignore
        data_sc = coords.SkyCoord(data_if, copy=False)

        return cls.from_format(data_sc, width=width, amplitude=amplitude, affine=affine, name=name, metadata=metadata)

    @from_format.register(coords.BaseRepresentation)
    @from_format.register(icoords.InterpolatedRepresentation)
    @classmethod
    def _from_format_representation(
        cls,
        data: coords.BaseRepresentation | icoords.InterpolatedRepresentation,
        width: Widths | None,
        amplitude: Callable | None = None,
        *,
        affine: u.Quantity,
        name: str | None = None,
        metadata: dict | None = None,
        frame: FrameLikeType,
        **kwargs: Any,
    ) -> Path:
        theframe = parse_framelike(frame)
        rep_type = data.__class__
        if "s" in data.differentials:
            dif_type = data.differentials["s"].__class__
        else:
            dif_type = None

        data_f = theframe.realize_frame(data, representation_type=rep_type, differential_type=dif_type)
        return cls.from_format(data_f, width=width, amplitude=amplitude, affine=affine, name=name, metadata=metadata)

    # ===============================================================
    # Interoperability

    def __array_function__(
        self, func: Callable[..., Any], types: tuple[type, ...], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        # LOCAL
        if func not in PATH_FUNCS:
            return NotImplemented

        finfo = PATH_FUNCS[func](self)
        if not finfo.validate_types(types):
            return NotImplemented
        return finfo.func(*args, **kwargs)

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
        if self.width is not None:
            width = self.width(affine) if not angular else self.width_angular(affine)
        else:
            width = None
        # TODO! add amplitude (density)
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

        # TODO! is there a more succinct Astropy func for this?
        r = self.data(affine).represent_as(coords.SphericalRepresentation)
        distance = r.distance.to_value(width.unit)

        return np.abs(np.arctan2(width.value, distance)) << u.rad

    # -----------------------------------------------------

    @overload
    def separation(
        self, point: CoordinateType, *, interpolate: Literal[True] = True, affine: u.Quantity | None = None
    ) -> IUSU:
        ...

    @overload
    def separation(
        self, point: CoordinateType, *, interpolate: Literal[False] = False, affine: u.Quantity | None = None
    ) -> coords.Angle:
        ...

    @format_doc(icoords.InterpolatedSkyCoord.separation.__doc__)
    def separation(
        self,
        point: CoordinateType,
        *,
        interpolate: bool = True,
        affine: u.Quantity | None = None,
    ) -> coords.Angle | IUSU:
        return self.data.separation(point, interpolate=interpolate, affine=affine)

    # -------------------------------------------

    @overload
    def separation_3d(
        self,
        point: CoordinateType,
        *,
        interpolate: Literal[True] = True,
        affine: u.Quantity | None = None,
    ) -> IUSU:
        ...

    @overload
    def separation_3d(
        self,
        point: CoordinateType,
        *,
        interpolate: Literal[False] = False,
        affine: u.Quantity | None = None,
    ) -> coords.Distance:
        ...

    @format_doc(icoords.InterpolatedSkyCoord.separation_3d.__doc__)
    def separation_3d(
        self,
        point: CoordinateType,
        *,
        interpolate: bool = True,
        affine: u.Quantity | None = None,
    ) -> coords.Distance | IUSU:
        return self.data.separation_3d(point, interpolate=interpolate, affine=affine)

    # -----------------------------------------------------

    def _closest_res_to_point(
        self, point: CoordinateType, *, angular: bool = False, affine: u.Quantity | None = None
    ) -> OptimizeResult:
        """Closest to stream, ignoring width"""
        if angular:
            sep_fn = self.separation(point, interpolate=True, affine=affine)
        else:
            sep_fn = self.separation_3d(point, interpolate=True, affine=affine)

        afn = self.affine if affine is None else affine
        res: OptimizeResult = minimize_scalar(
            lambda afn: sep_fn(afn).value,
            bounds=[afn.value.min(), afn.value.max()],
        )
        return res

    def closest_affine_to_point(
        self, point: CoordinateType, *, angular: bool = False, affine: u.Quantity | None = None
    ) -> u.Quantity:
        """Closest affine, ignoring width"""
        afn = self.affine if affine is None else affine
        res = self._closest_res_to_point(point, angular=angular, affine=afn)
        pt_afn = res.x << afn.unit
        return pt_afn

    def closest_position_to_point(
        self, point: CoordinateType, *, angular: bool = False, affine: u.Quantity | None = None
    ) -> coords.SkyCoord:
        """Closest point, ignoring width"""
        return self.position(self.closest_affine_to_point(point, angular=angular, affine=affine))


##############################################################################
# NumPy Overloading

PATH_FUNCS = NumPyOverloader(default_dispatch_on=Path)


@PATH_FUNCS.implements(np.concatenate, dispatch_on=Path, types=Path)
def concatenate(
    seqpaths: Sequence[Path],
    axis: int = 0,
    out: Path | None = None,
    dtype: np.dtype | None = None,
    casting: str = "same_kind",
) -> Path:
    # ----------------------------------------
    # Validation

    N = len(seqpaths)
    if N == 0:
        raise ValueError("need at least one array to concatenate")
    elif N > 2:
        raise ValueError("cannot concatenate more than 2 paths")

    if out is not None:
        raise ValueError("out must be None")
    elif dtype is not None:
        raise ValueError("dtype must be None")

    if N == 1:
        if axis != 0:
            raise ValueError("axis must be 0 for 1 path")
        return seqpaths[0]
    # else:  N == 2

    # ----------------------------------------

    npth, ppth = seqpaths
    if npth.frame != ppth.frame:
        raise ValueError("the paths must have the same frame")
    elif npth.frame.representation_type != ppth.frame.representation_type:
        raise ValueError("the paths must have the same representation_type")

    affine = cast(u.Quantity, np.concatenate((-npth.affine[::-1], ppth.affine)))

    # get representations, uninterpolated
    # TODO! add concatenated to InterpolatedSkyCoord
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
        raise ValueError("negow and posow is None")
    else:
        uninterp_width = np.concatenate((negow.uninterpolated[::-1], posow.uninterpolated))
        uninterp_width = cast(BaseWidth, uninterp_width)
        width = InterpolatedWidths.from_format(uninterp_width, affine=affine)

    # Name
    name = ppth.name if (npth.name == ppth.name) else f"{npth.name} | {ppth.name}"

    # Metadata
    # TODO! a better merge
    # metadata = merge(npth.meta, ppth.meta, metadata_conflicts="warn")
    metadata = {"npth": npth.meta, "ppth": ppth.meta}

    # TODO! amplitude
    return Path(c, width=width, amplitude=None, name=name, metadata=metadata)


# ============================================================================


@get_frame.register
def _get_frame_path(path: Path) -> coords.BaseCoordinateFrame:
    return path.frame
