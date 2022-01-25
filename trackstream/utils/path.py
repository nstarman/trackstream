# -*- coding: utf-8 -*-

"""Path are an affine-parameterized path."""

__all__ = ["Path", "path_moments"]


##############################################################################
# IMPORTS

# STDLIB
import copy
import typing as T

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
from astropy.coordinates import BaseCoordinateFrame, SkyCoord
from astropy.utils.decorators import format_doc
from scipy.optimize import OptimizeResult, minimize_scalar

# LOCAL
from .interpolate import InterpolatedUnivariateSplinewithUnits as IUSU
from .interpolated_coordinates import InterpolatedCoordinateFrame, InterpolatedSkyCoord
from trackstream._type_hints import CoordinateType, FrameLikeType
from trackstream.utils import resolve_framelike

##############################################################################
# CODE
##############################################################################


class path_moments(T.NamedTuple):
    mean: SkyCoord
    width: u.Quantity


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

    _name: T.Optional[str]
    _frame: coord.BaseCoordinateFrame
    _original_path: T.Any
    _iscrd: InterpolatedSkyCoord
    _original_width: T.Union[u.Quantity, T.Callable, None]
    _width_fn: T.Optional[T.Callable]
    _amplitude_fn: T.Optional[T.Callable]

    def __init__(
        self,
        path: T.Union[InterpolatedCoordinateFrame, InterpolatedSkyCoord],
        width: T.Union[u.Quantity, T.Callable, None] = None,  # func(affine)
        amplitude: T.Union[u.Quantity, T.Callable, None] = None,  # FIXME!
        *,
        name: T.Optional[str] = None,
        affine: T.Optional[u.Quantity] = None,
        frame: T.Optional[FrameLikeType] = None,
    ) -> None:
        self._name = str(name) if name is not None else name

        # Frame
        if frame is None:
            # unless `path` has a frame (is not `BaseRepresentation`).
            if isinstance(path, BaseCoordinateFrame):
                frame = path.replicate_without_data()
            elif isinstance(path, SkyCoord):  # SkyCoord & related
                frame = path.frame.replicate_without_data()
            # else: pass  # path = Representation handled in resolve_framelike.
        self._frame = resolve_framelike(frame)  # (an instance, not class)

        # --------------
        # path

        self._original_path = path.copy()  # original path, for safekeeping.

        # options are: BaseRepresentation, InterpolatedRepresentation
        #              BaseCoordinateFrame, InterpolatedCoordinateFrame
        #              SkyCoord, InterpolatedSkyCoord
        # need to end up with a InterpolatedSkyCoord
        # TODO! make fail if wrong type
        if isinstance(path, coord.BaseRepresentation):  # works for interp
            path = self.frame.realize_frame(path)

        if isinstance(path, InterpolatedCoordinateFrame):
            pass  # TODO! combine this with the below
        elif isinstance(path, coord.BaseCoordinateFrame):
            path = InterpolatedCoordinateFrame(path, affine=affine)

        path = InterpolatedSkyCoord(path, affine=affine)
        self._iscrd = path.transform_to(self.frame)

        # --------------
        # Width
        # this needs to be in physical coordinates
        # the initialization is separated out so that base classes can pass
        # `None` here and outside do stuff like have angular widths.

        self._original_width = None
        self._width_fn = None

        if width is not None:
            self._initialize_width(path, width)

    @property
    def name(self) -> T.Optional[str]:
        return self._name

    @property  # read-only
    def frame(self) -> coord.BaseCoordinateFrame:
        """The preferred frame (instance) of the Path."""
        return self._frame

    @property
    def data(self) -> InterpolatedSkyCoord:
        """The path, protected."""
        return self._iscrd

    @property
    def affine(self) -> u.Quantity:
        """Affine parameter along ``path``."""
        return self.data.affine

    # -----------------------------------------------------

    def _initialize_width(
        self,
        path: InterpolatedSkyCoord,
        width: T.Union[u.Quantity, T.Callable],
    ) -> None:
        # TODO clean this up and stuff
        # this is separated out so that base classes
        # can do stuff like have angular widths

        if callable(width):
            # just testing
            _ws = width(path.affine)
            if _ws.unit.physical_type != "length":
                raise ValueError("width must have units of length")

            o_w = width

        else:
            # clean
            o_w = u.Quantity(width, copy=False)

            # check
            if o_w.unit.physical_type != "length":
                raise ValueError("width must have units of length")

            # interpolate
            # first check if need to broadcast
            _ws = np.ones(len(path)) * o_w if o_w.isscalar else o_w
            width = IUSU(path.affine, _ws)

        self._original_width = copy.deepcopy(o_w)
        self._width_fn = width

    #################################################################
    # Math on the Track!

    def __call__(
        self, affine: T.Optional[u.Quantity] = None, *, angular: bool = False
    ) -> path_moments:
        """Call.

        Parameters
        ----------
        affine : `~astropy.units.Quantity` array-like or None, optional
            The affine interpolation parameter. If None (default), return
            all positions.
        angular : bool, optional keyword-only

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

        # TODO? add a directionality?
        # TODO allow for higher moments

        return path_moments(mean, width)

    # -----------------------

    def position(
        self,
        affine: T.Optional[u.Quantity] = None,
    ) -> T.Union[InterpolatedSkyCoord, SkyCoord]:
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

    def width(self, affine: T.Optional[u.Quantity] = None) -> u.Quantity:
        """Return the (1-sigma) width of the track at affine points.

        Parameters
        ----------
        affine : `~astropy.units.Quantity` array-like or None, optional
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
        return self._width_fn(u.Quantity(affine, copy=False))

    def width_angular(self, affine: T.Optional[u.Quantity] = None) -> u.Quantity:
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
        width = self.width(affine)
        if width.unit.physical_type == "angle":
            return width

        # TODO! is there a more succinct Astropy func for this?
        r = self.data(affine).represent_as(coord.SphericalRepresentation)
        distance = r.distance.to_value(width.unit)

        return np.abs(np.arctan2(width.value, distance)) << u.rad

    # -----------------------------------------------------

    @format_doc(InterpolatedSkyCoord.separation.__doc__)
    def separation(
        self,
        point: CoordinateType,
        *,
        interpolate: bool = True,
        affine: T.Optional[u.Quantity] = None,
    ) -> T.Union[coord.Angle, IUSU]:
        return self.data.separation(point, interpolate=interpolate, affine=affine)

    @format_doc(InterpolatedSkyCoord.separation_3d.__doc__)
    def separation_3d(
        self,
        point: CoordinateType,
        *,
        interpolate: bool = True,
        affine: T.Optional[u.Quantity] = None,
    ) -> T.Union[coord.Distance, IUSU]:
        return self.data.separation_3d(point, interpolate=interpolate, affine=affine)

    # -----------------------------------------------------

    def _closest_res_to_point(
        self, point: CoordinateType, *, angular: bool = False, affine: T.Optional[u.Quantity] = None
    ) -> OptimizeResult:
        """Closest to stream, ignoring width"""
        if angular:
            sep_fn = self.separation(point, interpolate=True, affine=affine)
        else:
            sep_fn = self.separation_3d(point, interpolate=True, affine=affine)

        affine = self.affine if affine is None else affine
        res: OptimizeResult = minimize_scalar(
            lambda afn: sep_fn(afn).value,
            bounds=[affine.value.min(), affine.value.max()],
        )
        return res

    def closest_affine_to_point(
        self, point: CoordinateType, *, angular: bool = False, affine: T.Optional[u.Quantity] = None
    ) -> u.Quantity:
        """Closest affine, ignoring width"""
        affine = self.affine if affine is None else affine
        res = self._closest_res_to_point(point, angular=angular, affine=affine)
        pt_affine = res.x << affine.unit
        return pt_affine

    def closest_position_to_point(
        self, point: CoordinateType, *, angular: bool = False, affine: T.Optional[u.Quantity] = None
    ) -> SkyCoord:
        """Closest point, ignoring width"""
        return self.position(self.closest_affine_to_point(point, angular=angular, affine=affine))


##############################################################################
# END
