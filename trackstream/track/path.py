# -*- coding: utf-8 -*-

"""Path are an affine-parameterized path."""

__all__ = ["Path", "path_moments"]


##############################################################################
# IMPORTS

# STDLIB
import copy
from typing import Any, Callable, Dict, NamedTuple, Optional, OrderedDict, Tuple, Type, Union, cast

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
from astropy.coordinates import BaseCoordinateFrame, BaseRepresentation, SkyCoord
from astropy.coordinates import concatenate as concatenate_coords
from astropy.units import Quantity, StructuredUnit
from astropy.utils.decorators import format_doc
from astropy.utils.metadata import MetaData, merge
from astropy.utils.misc import indent
from interpolated_coordinates import InterpolatedCoordinateFrame, InterpolatedSkyCoord
from interpolated_coordinates.utils import InterpolatedUnivariateSplinewithUnits as IUSU
from matplotlib.pyplot import Axes
from numpy import abs, arctan2, atleast_1d, concatenate, dtype, zeros
from numpy.lib.recfunctions import recursive_fill_fields
from scipy.optimize import OptimizeResult, minimize_scalar

# LOCAL
from .plot import plot_cov
from trackstream._type_hints import CoordinateType, FrameLikeType
from trackstream.base import CommonBase
from trackstream.utils.misc import is_structured
from trackstream.visualization import CLike, PlotDescriptorBase

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

    def in_path_frame(
        self,
        affine: Optional[Quantity] = None,
        *,
        c: CLike = "tab:blue",
        ax: Optional[Axes] = None,
        format_ax: bool = False,
        **kwargs: Any,
    ) -> Axes:
        path, _ax, *_ = self._setup(ax)
        kw = self._get_kw(kwargs, label=path.name)

        data = path.position(affine=affine)
        width = path.width(affine=affine)  # TODO! width_angular

        for i, w in enumerate(width):
            plot_cov((data.lon[i], data.lat[i]), std=w, facecolor="gray", alpha=0.5)

        _ax.fill_between(data.lon, data.lat - width["lat"], data.lat + width["lat"])
        _ax.plot(data.lon, data.lat, **kw)

        if format_ax:  # Axes settings
            _ax.set_xlabel(f"Lon (Stream) [{_ax.get_xlabel()}]", fontsize=13)
            _ax.set_ylabel(f"Lat (Stream) [{_ax.get_ylabel()}]", fontsize=13)
            _ax.grid(True)
            _ax.legend()

        return _ax


class Path(CommonBase):
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

    _name: Optional[str]
    _frame: coord.BaseCoordinateFrame
    _original_path: Any
    _iscrd: InterpolatedSkyCoord
    _original_width: Union[Quantity, Callable, None]
    _amplitude_fn: Optional[Callable]

    _width_interps: Optional[Dict[str, IUSU]]
    _width_dtype: Optional[dtype]
    _width_unit: Optional[StructuredUnit]

    _meta: dict
    meta = MetaData(copy=True)

    plot = PathPlotter()

    def __init__(
        self,
        path: Union[
            InterpolatedCoordinateFrame,
            InterpolatedSkyCoord,
            BaseRepresentation,
            SkyCoord,
            BaseCoordinateFrame,
        ],
        /,
        width: Union[Quantity, Callable, None] = None,  # func(affine)
        amplitude: Union[Quantity, Callable, None] = None,  # FIXME!
        *,
        name: Optional[str] = None,
        affine: Optional[Quantity] = None,
        frame: Optional[FrameLikeType] = None,
        representation_type: Optional[Type[BaseRepresentation]] = None,
        meta: Optional[dict] = None,
    ) -> None:
        self._name = str(name) if name is not None else name
        cast(OrderedDict, self.meta).update(meta or {})

        # Frame
        if frame is None:
            # unless `path` has a frame (is not `BaseRepresentation`).
            if isinstance(path, BaseCoordinateFrame):
                _frame = path.replicate_without_data()
                _frame.representation_type = path.representation_type
            elif isinstance(path, SkyCoord):  # SkyCoord & related
                _frame = path.frame.replicate_without_data()
                _frame.representation_type = path.representation_type
            else:  # path = Representation handled in resolve_framelike.
                _frame = frame
        else:
            _frame = frame

        super().__init__(
            frame=_frame, representation_type=representation_type, differential_type=None
        )
        # TODO differential_type

        # --------------
        # path

        self._original_path = path.copy()  # original path, for safekeeping.

        # options are: BaseRepresentation, InterpolatedRepresentation
        #              BaseCoordinateFrame, InterpolatedCoordinateFrame
        #              SkyCoord, InterpolatedSkyCoord
        # need to end up with a InterpolatedSkyCoord
        path_f: Union[
            InterpolatedCoordinateFrame,
            InterpolatedSkyCoord,
            SkyCoord,
            BaseCoordinateFrame,
        ]
        if isinstance(path, BaseRepresentation):  # works for interp
            path_f = self.frame.realize_frame(path, representation_type=representation_type)
        else:
            path_f = path

        path_if: Union[InterpolatedCoordinateFrame, InterpolatedSkyCoord, SkyCoord]
        if isinstance(path_f, BaseCoordinateFrame):
            path_if = InterpolatedCoordinateFrame(path_f, affine=affine)
        else:
            path_if = path_f

        path_isc = InterpolatedSkyCoord(
            path_if, affine=affine, representation_type=representation_type
        )
        self._iscrd = path_isc.transform_to(self.frame)
        self._iscrd.representation_type = self.representation_type
        # TODO! Differential

        # --------------
        # Width
        # this needs to be in physical coordinates
        # the initialization is separated out so that base classes can pass
        # `None` here and outside do stuff like have angular widths.

        self._original_width = None
        self._width_interps = self._width_dtype = self._width_unit = None

        if width is not None:
            if callable(width):
                self._initialize_callable_width(path_isc, width)
            else:
                self._initialize_width(path_isc, width)

    @property
    def name(self) -> Optional[str]:
        return self._name

    @property
    def data(self) -> InterpolatedSkyCoord:
        """The interpolated data."""
        return self._iscrd

    @property
    def affine(self) -> Quantity:
        """Affine parameter along ``path``."""
        return self.data.affine

    @property
    def _data_component_names(self) -> Dict[str, str]:
        """Return dict[frame name, rep name]."""
        cns: Dict[str, str] = self._iscrd.get_representation_component_names()
        if "s" in self._iscrd.data.data.differentials:  # add diff, if exists
            cns.update(self._iscrd.get_representation_component_names("s"))
        return cns

    # -----------------------------------------------------

    def _initialize_callable_width(
        self, path: InterpolatedSkyCoord, width: Callable[..., Quantity]
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
        self._original_width = copy.deepcopy(o_w)  # saving the original width

        if not is_structured(o_w):  # not structured
            raise NotImplementedError("TODO!")
        else:
            pass  # TODO! checks

            # # Check
            # if (pt := o_w_unit.physical_type) not in ("length", "angle"):
            #     raise ValueError(f"width must have units of length / angle, not {pt}")

        self._width_interps = {n: IUSU(iscrd.affine, o_w[n]) for n in o_w.dtype.names}
        self._width_dtype = o_w.dtype
        self._width_unit = cast(StructuredUnit, o_w.unit)

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

    #################################################################
    # Math on the Track!

    def __call__(self, affine: Optional[Quantity] = None, *, angular: bool = False) -> path_moments:
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
        affine: Optional[Quantity] = None,
    ) -> Union[InterpolatedSkyCoord, SkyCoord]:
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

    def width(self, affine: Optional[Quantity] = None) -> Quantity:
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

    def width_angular(self, affine: Optional[Quantity] = None) -> Quantity:
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
        affine: Optional[Quantity] = None,
    ) -> Union[coord.Angle, IUSU]:
        return self.data.separation(point, interpolate=interpolate, affine=affine)

    @format_doc(InterpolatedSkyCoord.separation_3d.__doc__)
    def separation_3d(
        self,
        point: CoordinateType,
        *,
        interpolate: bool = True,
        affine: Optional[Quantity] = None,
    ) -> Union[coord.Distance, IUSU]:
        return self.data.separation_3d(point, interpolate=interpolate, affine=affine)

    # -----------------------------------------------------

    def _closest_res_to_point(
        self, point: CoordinateType, *, angular: bool = False, affine: Optional[Quantity] = None
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
        self, point: CoordinateType, *, angular: bool = False, affine: Optional[Quantity] = None
    ) -> Quantity:
        """Closest affine, ignoring width"""
        afn = self.affine if affine is None else affine
        res = self._closest_res_to_point(point, angular=angular, affine=afn)
        pt_afn = res.x << afn.unit
        return pt_afn

    def closest_position_to_point(
        self, point: CoordinateType, *, angular: bool = False, affine: Optional[Quantity] = None
    ) -> SkyCoord:
        """Closest point, ignoring width"""
        return self.position(self.closest_affine_to_point(point, angular=angular, affine=affine))


# ----------------------------------------------------------------------------


def concatenate_paths(
    paths: Tuple[Path, Path], /, *, name: Optional[str] = None, metadata_conflicts: str = "warn"
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
    elif npth.representation_type != ppth.representation_type:
        raise ValueError("the paths must have the same representation_type")

    # TODO! should it be original_path and _original_width?
    affine = concatenate((-npth.affine[::-1], ppth.affine))
    c = concatenate_coords((npth._original_path[::-1], ppth._original_path))

    negow = npth._original_width
    posow = ppth._original_width
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
        width=width,
        amplitude=None,  # FIXME!
        # keywords
        name=name,
        affine=affine,
        frame=ppth.frame,
        representation_type=ppth.representation_type,
        meta=meta,
    )
