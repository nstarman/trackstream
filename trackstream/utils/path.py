# -*- coding: utf-8 -*-

"""Path are an affine-parameterized path."""

__all__ = ["Path", "path_moments"]


##############################################################################
# IMPORTS

# STDLIB
import copy
from typing import Any, Callable, NamedTuple, Optional, Tuple, Union, cast
from typing import Type

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
from astropy.coordinates import BaseCoordinateFrame, SkyCoord, BaseRepresentation
from astropy.coordinates import concatenate as concatenate_coords
from astropy.units import Quantity
from astropy.utils.decorators import format_doc
from interpolated_coordinates import InterpolatedCoordinateFrame, InterpolatedSkyCoord
from interpolated_coordinates.utils import InterpolatedUnivariateSplinewithUnits as IUSU
from scipy.optimize import OptimizeResult, minimize_scalar
from astropy.utils.misc import indent

# LOCAL
from trackstream._type_hints import CoordinateType, FrameLikeType
from trackstream.base import CommonBase

##############################################################################
# CODE
##############################################################################


class path_moments(NamedTuple):
    mean: SkyCoord
    width: Quantity


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

        Must be in physical distance.
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
        If None (default), taken from the config (``conf.default_frame``)
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
    _width_fn: Optional[Callable]
    _amplitude_fn: Optional[Callable]

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
    ) -> None:
        self._name = str(name) if name is not None else name

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

        super().__init__(frame=_frame, representation_type=representation_type)

        # --------------
        # path

        self._original_path = path.copy()  # original path, for safekeeping.

        # options are: BaseRepresentation, InterpolatedRepresentation
        #              BaseCoordinateFrame, InterpolatedCoordinateFrame
        #              SkyCoord, InterpolatedSkyCoord
        # need to end up with a InterpolatedSkyCoord
        path_f: Union[
            InterpolatedCoordinateFrame, InterpolatedSkyCoord, SkyCoord, BaseCoordinateFrame
        ]
        if isinstance(path, BaseRepresentation):  # works for interp
            path_f = self.frame.realize_frame(path)
            path_f.representation_type = self.representation_type
        else:
            path_f = path

        path_if: Union[InterpolatedCoordinateFrame, InterpolatedSkyCoord, SkyCoord]
        if isinstance(path_f, BaseCoordinateFrame):
            path_if = InterpolatedCoordinateFrame(path_f, affine=affine)
        else:
            path_if = path_f

        path_isc = InterpolatedSkyCoord(path_if, affine=affine)
        self._iscrd = path_isc.transform_to(self.frame)
        self._iscrd.representation_type = self.representation_type

        # --------------
        # Width
        # this needs to be in physical coordinates
        # the initialization is separated out so that base classes can pass
        # `None` here and outside do stuff like have angular widths.

        self._original_width = None
        self._width_fn = None

        if width is not None:
            self._initialize_width(path_isc, width)

    @property
    def name(self) -> Optional[str]:
        return self._name

    @property
    def data(self) -> InterpolatedSkyCoord:
        """The path, protected."""
        return self._iscrd

    @property
    def affine(self) -> Quantity:
        """Affine parameter along ``path``."""
        return self.data.affine

    # -----------------------------------------------------

    def _initialize_width(
        self,
        path: InterpolatedSkyCoord,
        width: Union[Quantity, Callable],
    ) -> None:
        """Initialize the width function."""
        if callable(width):
            # Check
            _ws = width(path.affine)
            if _ws.unit.physical_type != "length":
                raise ValueError("width must have units of length")
            o_w = cast(Quantity, width)

        else:
            # Clean
            o_w = Quantity(width, copy=False)
            o_w_unit = cast(u.UnitBase, o_w.unit)

            # Check
            if o_w_unit.physical_type != "length":
                raise ValueError("width must have units of length")

            # Interpolate (first check if need to broadcast)
            _ws = cast(Quantity, np.ones(len(path)) * o_w) if o_w.isscalar else o_w
            width = IUSU(path.affine, _ws)

        self._original_width = copy.deepcopy(o_w)
        self._width_fn = width

    def __repr__(self) -> str:
        rs = []

        # 0) header (standard repr)
        header: str = object.__repr__(self)
        rs.append(header)

        # 1) name
        name = str(self.name)
        rs.append("  Name: " + name)

        # 2) data
        rs.append(indent(repr(self.data), width=2))

        return "\n".join(rs)

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
        if self._width_fn is None:
            raise ValueError("Path does not have a defined width.")
        affine = self.affine if affine is None else affine
        return self._width_fn(affine)

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

        return np.abs(np.arctan2(width.value, distance)) << u.rad

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


def concatenate_paths(paths: Tuple[Path, Path]) -> Path:
    """Concatenate `trackstream.utils.path.Path` instances.

    Parameters
    ----------
    paths : tuple[`trackstream.utils.path.Path`, `trackstream.utils.path.Path`]

    Returns
    -------
    `trackstream.utils.path.Path`

    Raises
    ------
    TypeError
        if ``_original_width`` on either path is not a |Quantity|.
    """
    # TODO! Even better is to override __array_function_ so can use np.concatenate
    neg_path, pos_path = paths
    if neg_path.frame != pos_path.frame:
        raise ValueError

    # TODO! should it be original_path and _original_width?
    affine = np.concatenate((-neg_path.affine[::-1], pos_path.affine))
    c = concatenate_coords((neg_path._original_path[::-1], pos_path._original_path))

    negow = neg_path._original_width
    posow = pos_path._original_width
    if not isinstance(negow, Quantity) or not isinstance(posow, Quantity):
        raise TypeError
    else:
        negow = cast(Quantity, negow)
        posow = cast(Quantity, posow)
    sigma = np.concatenate((negow[::-1], posow))

    return Path(c, width=sigma, affine=affine, frame=pos_path.frame)
