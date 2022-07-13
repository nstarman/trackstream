"""Path are an affine-parameterized path."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import copy
from typing import Any, Callable, NamedTuple, cast

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
from astropy.coordinates import (
    BaseCoordinateFrame,
    BaseDifferential,
    BaseRepresentation,
    SkyCoord,
    concatenate_representations,
)
from astropy.units import Quantity, StructuredUnit
from astropy.utils.decorators import format_doc
from astropy.utils.metadata import MetaData, merge
from astropy.utils.misc import indent
from attrs import converters, define, field
from interpolated_coordinates import (
    InterpolatedCoordinateFrame,
    InterpolatedRepresentation,
    InterpolatedSkyCoord,
)
from interpolated_coordinates.utils import InterpolatedUnivariateSplinewithUnits as IUSU
from numpy import abs, arctan2, atleast_1d, concatenate, dtype, zeros
from numpy.lib.recfunctions import recursive_fill_fields
from scipy.optimize import OptimizeResult, minimize_scalar

# LOCAL
from trackstream._typing import CoordinateType, FrameLikeType
from trackstream.base import FramedBase
from trackstream.utils.coord_utils import resolve_framelike
from trackstream.utils.misc import is_structured
from trackstream.visualization import PlotDescriptorBase

__all__ = ["Path", "path_moments"]


##############################################################################
# PARAMETERS


class path_moments(NamedTuple):
    mean: SkyCoord
    width: Quantity


##############################################################################
# CODE
##############################################################################


class PathPlotter(PlotDescriptorBase["Path"]):
    """Plot descriptor for a Path."""


@define(frozen=True)
class Path(FramedBase):
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
    plot = PathPlotter()

    data: InterpolatedSkyCoord = field()
    _width: Quantity | Callable | None = None
    _amplitude: Callable | None = None

    name: str | None = field(default=None, kw_only=True, converter=converters.optional(str))
    _meta: dict = field(factory=dict, kw_only=True)

    frame: BaseCoordinateFrame = field(init=False, kw_only=True, converter=resolve_framelike)
    frame_representation_type: type[BaseRepresentation] = field(init=False, kw_only=True)
    frame_differential_type: type[BaseDifferential] | None = field(init=False, kw_only=True)

    _width_interps: dict[str, IUSU] | None = field(init=False, default=None, repr=False)
    _width_dtype: dtype | None = field(init=False, default=None, repr=False)
    _width_unit: StructuredUnit | None = field(init=False, default=None, repr=False)
    _original_width: Quantity | Callable | None = field(init=False, repr=False)

    _original_path: Any = field(init=False, repr=False)

    @frame.default
    def _frame_factory(self):
        data = self.data
        frame = resolve_framelike(data)
        return frame

    @frame_representation_type.default  # type: ignore
    def _frame_representation_type_default(self):
        return self.frame.representation_type

    @frame_differential_type.default  # type: ignore
    def _frame_differential_type_default(self):
        return self.frame.differential_type

    @_original_path.default
    def _original_path_factory(self):
        return self.data

    @data.validator
    def _data_validator(self, _, value):
        if not isinstance(value, InterpolatedSkyCoord):
            raise TypeError("use from_<X> instead")

    def __attrs_post_init__(self):
        super().__attrs_post_init__()

        if self._width is not None:
            if callable(self._width):
                self._initialize_callable_width(self.data, self._width)
            else:
                self._initialize_width(self.data, self._width)

    @classmethod
    def from_skycoord(
        cls,
        data: SkyCoord,
        width: Quantity | Callable | None,
        amplitude: Callable | None = None,
        *,
        affine: Quantity,
        name: str | None = None,
        meta: dict | None = None,
    ):
        if isinstance(data, InterpolatedSkyCoord) and data.affine != affine:
            raise ValueError
        isc = InterpolatedSkyCoord(data, affine=affine)
        return cls(
            isc,
            width=width,  # type: ignore
            amplitude=amplitude,  # type: ignore
            name=name,
            meta=meta or {},  # type: ignore
        )

    @classmethod
    def from_frame(
        cls,
        data: BaseCoordinateFrame | InterpolatedCoordinateFrame,
        width: Quantity | Callable | None,
        amplitude: Callable | None = None,
        *,
        affine: Quantity,
        name: str | None = None,
        meta: dict | None = None,
    ):
        if isinstance(data, InterpolatedCoordinateFrame) and data.affine != affine:
            raise ValueError

        data_if = InterpolatedCoordinateFrame(data, affine=affine)  # type: ignore
        data_sc = SkyCoord(data_if, copy=False)

        return cls.from_skycoord(data_sc, width=width, amplitude=amplitude, affine=affine, name=name, meta=meta)

    @classmethod
    def from_representation(
        cls,
        data: BaseRepresentation | InterpolatedRepresentation,
        width: Quantity | Callable | None,
        amplitude: Callable | None = None,
        *,
        frame: FrameLikeType,
        affine: Quantity,
        name: str | None = None,
        meta: dict | None = None,
    ):
        theframe = resolve_framelike(frame)
        rep_type = data.__class__
        if "s" in data.differentials:
            dif_type = data.differentials["s"].__class__
        else:
            dif_type = None

        data_f = theframe.realize_frame(data, representation_type=rep_type, differential_type=dif_type)
        return cls.from_frame(data_f, width=width, amplitude=amplitude, affine=affine, name=name, meta=meta)

    @property
    def affine(self) -> Quantity:
        """Affine parameter along ``path``."""
        return self.data.affine

    @property
    def _data_component_names(self) -> dict[str, str]:
        """Return dict[frame name, rep name]."""
        cns: dict[str, str] = self.data.get_representation_component_names()
        if "s" in self.data.data.data.differentials:  # add diff, if exists
            cns.update(self.data.get_representation_component_names("s"))
        return cns

    # -----------------------------------------------------

    def _initialize_callable_width(
        self,
        path: InterpolatedSkyCoord,
        width: Callable[..., Quantity],
    ) -> None:
        raise NotImplementedError("TODO!")

        # # Check
        # _ws = width(path.affine)
        # if (pt := _ws.unit.physical_type) not in ("length", "angle"):
        #     raise ValueError(f"width must have units of length / angle, not {pt}")
        # o_w = cast(Quantity, width)

    def _initialize_width(self, iscrd: InterpolatedSkyCoord, width: Quantity) -> None:
        """Initialize the width function.

        Parameters
        ----------
        iscrd : (N,) InterpolatedSkyCoord
            The path mean.
        width : scalar or (N,) or (N, <=D) Quantity
            The width of the path. There are 3 options:

            1. `width` is a scalar or (N,) non-structured array:

                - If `path` is in angular coordinates `width` is the projected
                width on-sky. `width` of angular dimensions is easy, `width` of
                length dimensions is converted to angular dimensions using the
                distance of the `path` (and will error if missing).

                - If `path` is in cartesian dimensions `width` is the width in
                each dimension. Angular dimensions are converted using the
                distance.

            2. `width` is an (N, <=D) Quantity
                `width` is made into a structured Quantity with field names from
                and ordered by the `path` components. See the next option.

            3. `width` is a structured Quantity
                The field names of `width` are matched to the coordinate
                dimension names of `path`. Missing coordinates have 0 width.
        """
        # Setup
        o_w = Quantity(width, copy=False)
        object.__setattr__(self, "_original_width", copy.deepcopy(o_w))
        # saving the original width

        if not is_structured(o_w):  # not structured
            raise NotImplementedError("TODO!")
        else:
            pass  # TODO! checks

            # # Check
            # if (pt := o_w_unit.physical_type) not in ("length", "angle"):
            #     raise ValueError(f"width must have units of length / angle, not {pt}")

        _width_interps = {n: IUSU(iscrd.affine, o_w[n]) for n in o_w.dtype.names}
        object.__setattr__(self, "_width_interps", _width_interps)
        object.__setattr__(self, "_width_dtype", o_w.dtype)
        object.__setattr__(self, "_width_unit", cast(StructuredUnit, o_w.unit))

    def __repr__(self) -> str:
        r = ""

        # 0) header (standard repr)
        header: str = object.__repr__(self)
        r += header

        # 1) name
        r += f"\n  Name: {self.name}"

        # 2) data
        r += "\n" + indent(repr(self.data), width=2)

        return r

    def __len__(self) -> int:
        return len(self.data)

    #################################################################
    # Math on the Track!

    def __call__(self, affine: Quantity | None = None, *, angular: bool = False) -> path_moments:
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
        width = self.width(affine) if not angular else self.width_angular(affine)
        # TODO! add amplitude (density)
        return path_moments(mean, width)

    # -----------------------

    def position(
        self,
        affine: Quantity | None = None,
    ) -> InterpolatedSkyCoord | SkyCoord:
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

    def width(self, affine: Quantity | None = None) -> Quantity:
        """Return the (1-sigma) width of the track at affine points.

        Parameters
        ----------
        affine : `~astropy.units.Quantity` or None, optional
            The affine interpolation parameter. If None (default), return
            width evaluated at all "tick" interpolation points.

        Returns
        -------
        `~astropy.unitss.Quantity`
            Path width evaluated at ``affine``.
        """
        if self._width_interps is None:
            raise ValueError("Path does not have a defined width.")
        affine = self.affine if affine is None else cast(Quantity, atleast_1d(affine))

        ws = {k: v(affine) for k, v in self._width_interps.items()}
        out = zeros(affine.shape, dtype=self._width_dtype)
        recursive_fill_fields(ws, out)  # strips units
        # TODO! astropy override then don't need self._width_unit

        return Quantity(out, self._width_unit)

    def width_angular(self, affine: Quantity | None = None) -> Quantity:
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
        width: Quantity = self.width(affine)
        width_u = cast(u.UnitBase, width.unit)
        if width_u.physical_type == "angle":
            return width

        # TODO! is there a more succinct Astropy func for this?
        r = self.data(affine).represent_as(coord.SphericalRepresentation)
        distance = r.distance.to_value(width_u)

        return abs(arctan2(width.value, distance)) << u.rad

    # -----------------------------------------------------

    @format_doc(InterpolatedSkyCoord.separation.__doc__)
    def separation(
        self,
        point: CoordinateType,
        *,
        interpolate: bool = True,
        affine: Quantity | None = None,
    ) -> coord.Angle | IUSU:
        return self.data.separation(point, interpolate=interpolate, affine=affine)

    @format_doc(InterpolatedSkyCoord.separation_3d.__doc__)
    def separation_3d(
        self,
        point: CoordinateType,
        *,
        interpolate: bool = True,
        affine: Quantity | None = None,
    ) -> coord.Distance | IUSU:
        return self.data.separation_3d(point, interpolate=interpolate, affine=affine)

    # -----------------------------------------------------

    def _closest_res_to_point(
        self, point: CoordinateType, *, angular: bool = False, affine: Quantity | None = None
    ) -> OptimizeResult:
        """Closest to stream, ignoring width"""
        if angular:
            sep_fn = self.separation(point, interpolate=True, affine=affine)
        else:
            sep_fn = self.separation_3d(point, interpolate=True, affine=affine)
        sep_fn = cast(IUSU, sep_fn)

        afn = self.affine if affine is None else affine
        res: OptimizeResult = minimize_scalar(
            lambda afn: sep_fn(afn).value,
            bounds=[afn.value.min(), afn.value.max()],
        )
        return res

    def closest_affine_to_point(
        self, point: CoordinateType, *, angular: bool = False, affine: Quantity | None = None
    ) -> Quantity:
        """Closest affine, ignoring width"""
        afn = self.affine if affine is None else affine
        res = self._closest_res_to_point(point, angular=angular, affine=afn)
        pt_afn = res.x << afn.unit
        return pt_afn

    def closest_position_to_point(
        self, point: CoordinateType, *, angular: bool = False, affine: Quantity | None = None
    ) -> SkyCoord:
        """Closest point, ignoring width"""
        return self.position(self.closest_affine_to_point(point, angular=angular, affine=affine))


# ----------------------------------------------------------------------------


def concatenate_paths(
    paths: tuple[Path, Path], /, *, name: str | None = None, metadata_conflicts: str = "warn"
) -> Path:
    """Concatenate `trackstream.utils.path.Path` instances.

    Parameters
    ----------
    paths : (`Path`, `Path`),  positional-only

    name : str or None, optional keyword-only
        The name of the concatenated path. If `None` (default) the name will be
        a concatenation of the paths' names, unless they are equal in which case
        the name is the same.
    metadata_conflicts : str, optional keyword-only
        See `astropy.utils.metadata.merge`.

    Returns
    -------
    `trackstream.utils.path.Path`

    Raises
    ------
    TypeError
        if ``_original_width`` on either path is not a |Quantity|.
    """
    # TODO! Even better is to override __array_function_ so can use concatenate
    npth, ppth = paths
    if npth.frame != ppth.frame:
        raise ValueError("the paths must have the same frame")
    elif npth.frame_representation_type != ppth.frame_representation_type:
        raise ValueError("the paths must have the same representation_type")

    # TODO! should it be original_path and _original_width?
    affine = concatenate((-npth.affine[::-1], ppth.affine))

    # get representations, uninterpolated
    nr = npth.data.data.data
    if "s" in nr.differentials:
        nr.differentials["s"] = nr.differentials["s"].data
    pr = ppth.data.data.data
    if "s" in pr.differentials:
        pr.differentials["s"] = pr.differentials["s"].data
    r = concatenate_representations((nr[::-1], pr))
    c = InterpolatedSkyCoord(npth.data.realize_frame(r, affine=affine))

    negow = npth._width
    posow = ppth._width
    if not isinstance(negow, Quantity) or not isinstance(posow, Quantity):
        raise TypeError
    else:
        negow = cast(Quantity, negow)
        posow = cast(Quantity, posow)
    width = concatenate((negow[::-1], posow))

    # Name
    if name is not None:
        pass
    elif npth.name == ppth.name:
        name = ppth.name
    else:
        name = f"{npth.name} | {ppth.name}"

    # Metadata  # TODO! mergestrategy
    meta = merge(npth.meta, ppth.meta, metadata_conflicts=metadata_conflicts)

    return Path(
        c,
        # (kw)arg
        width=width,  # type: ignore
        amplitude=None,  # type: ignore  # FIXME!
        # keywords
        name=name,
        meta=meta,  # type: ignore
    )
