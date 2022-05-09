# -*- coding: utf-8 -*-

"""Fit a Rotated reference frame."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from copy import deepcopy
from types import FunctionType, MappingProxyType, MethodType
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple, Type, TypedDict, TypeVar
from typing import Union

# THIRD PARTY
import astropy.units as u
import scipy.optimize as opt
from astropy.coordinates import BaseCoordinateFrame, BaseRepresentation
from astropy.coordinates import CartesianRepresentation as CartRep
from astropy.coordinates import SkyCoord, SkyOffsetFrame, SphericalCosLatDifferential
from astropy.coordinates import UnitSphericalRepresentation as USphrRep
from astropy.units import Quantity
from astropy.utils.misc import indent
from erfa import ufunc as erfa_ufunc
from matplotlib.pyplot import Axes
from numpy import abs, array, average, column_stack, dot, linspace, median, ndarray, sqrt, square
from numpy import sum

# LOCAL
from trackstream._type_hints import EllipsisType, FrameLikeType
from trackstream.base import CommonBase
from trackstream.utils.coord_utils import reference_to_skyoffset_matrix
from trackstream.visualization import PlotDescriptorBase

if TYPE_CHECKING:
    # LOCAL
    from trackstream.stream import Stream

__all__ = ["RotatedFrameFitter", "residual"]


##############################################################################
# PARAMETERS

FT = TypeVar("FT", MethodType, FunctionType)


class _RotatedFrameFitterOptions(TypedDict):
    fix_origin: bool
    leastsquares: bool
    align_v: bool


##############################################################################
# CODE
##############################################################################


def cartesian_model(
    data: CartRep,
    *,
    lon: Union[Quantity, float],
    lat: Union[Quantity, float],
    rotation: Union[Quantity, float],
) -> Tuple[Quantity, Quantity, Quantity]:
    """Model from Cartesian Coordinates.

    Parameters
    ----------
    data : |CartesianRep|
        Cartesian representation of the data.
    lon, lat : float or |Angle| or |Quantity| instance
        The longitude and latitude origin for the reference frame.
        If float, assumed degrees.
    rotation : float or |Angle| or |Quantity| instance
        The final rotation of the frame about the ``origin``. The sign of
        the rotation is the left-hand rule.  That is, an object at a
        particular position angle in the un-rotated system will be sent to
        the positive latitude (z) direction in the final frame.
        If float, assumed degrees.

    Returns
    -------
    r, lat, lon : array_like
        Same shape as `x`, `y`, `z`.
    """
    rot_matrix = reference_to_skyoffset_matrix(lon, lat, rotation)
    rot_xyz: Quantity = dot(rot_matrix, data.xyz).T

    # cartesian to spherical
    r: Quantity = sqrt(sum(square(rot_xyz), axis=-1))
    _lon, _lat = erfa_ufunc.c2s(rot_xyz)

    return r, _lon, _lat


# -------------------------------------------------------------------


def residual(
    variables: Tuple[float, float, float], data: CartRep, scalar: bool = False
) -> Union[float, ndarray]:
    r"""How close phi2, the rotated latitude (dec), is to flat.

    Parameters
    ----------
    variables : tuple[float, float, float]
        (rotation, lon, lat)

        - rotation angle : float
            The final rotation of the frame about the ``origin``. The sign of
            the rotation is the left-hand rule.  That is, an object at a
            particular position angle in the un-rotated system will be sent to
            the positive latitude (z) direction in the final frame.
            In degrees.
        - lon, lat : float
            In degrees. If ICRS, equivalent to ra & dec.
    data : |CartesianRep|
        eg. ``ICRS.cartesian``

    Returns
    -------
    res : float or ndarray
        :math:`\rm{lat} - 0`.
        If `scalar` is True, then sum array_like to return float.

    Other Parameters
    ----------------
    scalar : bool, optional, keyword-only
        Whether to sum `res` into a float.
        Note that if `res` is also a float, it is unaffected.
    """
    rotation, lon, lat = variables

    _, _, phi2 = cartesian_model(data, lon=lon, lat=lat, rotation=rotation)
    # Residual
    res: ndarray = abs(phi2.to_value(u.deg) - 0.0) / len(phi2)  # phi2 - 0

    if scalar:
        sres: float = sum(res)
        return sres
    return res


#####################################################################


class RotatedFrameFitter(CommonBase):
    """Class to Fit Rotated Frames.

    The fitting is always on-sky.

    Parameters
    ----------
    origin : `~astropy.coordinates.SkyCoord`
        The location of point on sky about which to rotate.

    frame : `~astropy.coordinates.BaseCoordinateFrame` or None, optional keyword-only
        The frame. If `None` (default) uses the frame of `origin`.

    representation_type : `astropy.coordinates.BaseRepresentation` or None, optional keyword-only
        The representation type for the `frame`. If `None` (default) uses the
        current representation type of the `frame`. The fitting happens in
        `~astropy.coordinates.UnitSphericalRepresentation` but will be returned
        in this `representation_type`.

    Other Parameters
    ----------------
    rot_lower, rot_upper : |Quantity|, (optional, keyword-only)
        The lower and upper bounds in degrees. Default is (-180, 180] degree.
    origin_lim : |Quantity|, (optional, keyword-only)
        The symmetric lower and upper bounds on origin in degrees. Default is
        0.005 degree.

    fix_origin : bool, optional, keyword-only
        Whether to fix the origin point. Default is False.
    leastsquares : bool, optional, keyword-only
        Whether to to use :func:`~scipy.optimize.least_square` or
        :func:`~scipy.optimize.minimize`. Default is False.

    align_v : bool
        Whether to align by the velocity.
    """

    _origin: SkyCoord
    _bounds: ndarray
    _minimizer_kwargs: Dict[str, Any]

    def __init__(
        self,
        origin: SkyCoord,
        *,
        frame: Optional[FrameLikeType] = None,
        representation_type: Optional[Type[BaseRepresentation]] = None,
        **kwargs: Any,
    ) -> None:
        # Set the frame and representation_type
        super().__init__(
            frame=origin if frame is None else frame,
            representation_type=representation_type,
            differential_type=None,  # The differential type does not matter
        )

        # Origin.  Note the rep-type does not change the underlying data.
        self._origin = origin.transform_to(self.frame)
        self._origin.representation_type = self.representation_type

        # Create bounds
        bounds_args = ("rot_lower", "rot_upper", "origin_lim")
        bounds_kwargs = {k: kwargs.pop(k) for k in bounds_args if k in kwargs}
        self.set_bounds(**bounds_kwargs)

        # Process options
        self._default_options = _RotatedFrameFitterOptions(
            fix_origin=kwargs.pop("fix_origin", False),
            leastsquares=kwargs.pop("leastsquares", False),
            align_v=kwargs.pop("align_v", None),
        )

        # Minimizer kwargs are the leftovers
        self._minimizer_kwargs = kwargs

    @property
    def origin(self) -> SkyCoord:
        """The location of point on sky about which to rotate."""
        return self._origin

    @property
    def bounds(self) -> ndarray:
        """The fitting bounds."""
        return self._bounds

    @property
    def minimizer_kwargs(self) -> Dict[str, Any]:
        """The kwargs passed to the minimizer"""
        return self._minimizer_kwargs

    @property
    def default_fit_options(self) -> MappingProxyType:
        """The default fit options, including from initialization."""
        return MappingProxyType(dict(**self._default_options, **self.minimizer_kwargs))

    def __repr__(self) -> str:
        r = ""

        # 1) header (standard repr)
        header: str = object.__repr__(self)
        r += header

        # 2) Origin
        origin = repr(self.origin)
        r = "  Origin:"
        r += ("\n" + indent(origin)) if "\n" in origin else (" " + origin)

        # 3) Bounds
        bounds = repr(self.bounds)
        r = "  Bounds:"
        r += ("\n" + indent(bounds)) if "\n" in bounds else (" " + bounds)

        return r

    #######################################################

    def set_bounds(
        self,
        rot_lower: Quantity = Quantity(-180.0, u.deg),
        rot_upper: Quantity = Quantity(180.0, u.deg),
        origin_lim: Quantity = Quantity(0.005, u.deg),
    ) -> None:
        """Make bounds on Rotation parameter.

        Parameters
        ----------
        rot_lower, rot_upper : |Quantity|, optional
            The lower and upper bounds in degrees.
        origin_lim : |Quantity|, optional
            The symmetric lower and upper bounds on origin in degrees.
        """
        origin = self.origin.represent_as(USphrRep)

        rotation_bounds = (rot_lower.to_value(u.deg), rot_upper.to_value(u.deg))
        # longitude bounds (ra in ICRS).
        lon_bounds = (origin.lon + (-1, 1) * origin_lim).to_value(u.deg)
        # latitude bounds (dec in ICRS).
        lat_bounds = (origin.lat + (-1, 1) * origin_lim).to_value(u.deg)

        # stack bounds so rows are bounds.
        bounds = column_stack((rotation_bounds, lon_bounds, lat_bounds)).T

        self._bounds = bounds

    def align_v_positive_lon(
        self,
        data: SkyCoord,
        fit_values: Dict[str, Any],
        subsel: Union[EllipsisType, Sequence, slice] = Ellipsis,  # type: ignore
    ) -> Dict[str, Any]:
        """Align the velocity along the positive Longitudinal direction.

        Parameters
        ----------
        fit_values : dict
            The rotation and origin. Output of `~minimize`
        subsel : slice
            sub-select a portion of the `pm_lon_coslat` for determining
            the average velocity.

        Returns
        -------
        values : dict
            `fit_values` with "rotation" adjusted.
        """
        values = deepcopy(fit_values)  # copy for safety
        rotation = values["rotation"]

        # Make frame.
        frame = SkyOffsetFrame(**values)
        # Transform data and set dif-type to have ``pm_lon_coslat``.
        rot_data = data.transform_to(frame)
        rot_data.differential_type = SphericalCosLatDifferential

        # Get average velocity to determine whether need to rotate.  # TODO
        avg = median(rot_data.pm_lon_coslat[subsel])

        if avg < 0:  # need to flip
            rotation = rotation + Quantity(180, u.deg)

        return values

    #######################################################
    # Fitting

    def fit(
        self,
        data: SkyCoord,
        /,
        rot0: Optional[Quantity] = None,
        bounds: Optional[ndarray] = None,
        *,
        fix_origin: Optional[bool] = None,
        leastsquares: Optional[bool] = None,
        align_v: Optional[bool] = None,  # TODO!
        **minimizer_kwargs: Any,
    ) -> FrameOptimizeResult:
        """Find Best-Fit Rotated Frame.

        Parameters
        ----------
        data : `astropy.coordinates.SkyCoord`, positional only
        rot0 : Quantity, optional
            Initial guess for rotation
        bounds : array-like, optional
            Parameter bounds.
            ::
                [[rot_low, rot_up],
                 [lon_low, lon_up],
                 [lat_low, lat_up]]

        Returns
        -------
        res : Any
            The result of the minimization. Depends on arguments.
        dict[str, Any]
            Has fields "rotation" and "origin".

        Other Parameters
        ----------------
        fix_origin : bool, optional, keyword-only
            Whether to fix the origin.
        leastsquares : bool, optional, keyword-only
            Whether to to use :func:`~scipy.optimize.least_square` or
            :func:`~scipy.optimize.minimize` (default).
        align_v : bool, optional, keyword-only
            Whether to align velocity to be in positive direction
        fit_kwargs:
            Into whatever minimization package / function is used.
        """
        # Put data in right coordinates & cartesian representation but routed
        # through UnitSphericalRepresentation. The rotated frame is a spherical
        # representation, but the math is much easier with a Cartesian
        # representation.
        crd = data.transform_to(self.frame)
        rep: CartRep = crd.represent_as(USphrRep).represent_as(CartRep)

        # Put origin in right representation type
        origin: USphrRep = self.origin.represent_as(USphrRep)

        # -----------------------------
        # Prepare, using defaults for arguments not provided.

        # kwargs, preferring newer
        kwargs: Dict[str, Any] = {**self.minimizer_kwargs, **minimizer_kwargs}

        if rot0 is None:
            rot0 = kwargs.get("rot0", None)
            if rot0 is None:
                raise ValueError("no prespecified `rot0`; need to provide one.")

        bnds: ndarray = self.bounds if bounds is None else bounds

        # Origin
        if fix_origin is None:
            fix_origin = self._default_options["fix_origin"]
        if fix_origin:
            bnds[1, :] = average(bnds[1, :])
            bnds[2, :] = average(bnds[2, :])
            raise NotImplementedError("TODO")

        # Process fit options
        if leastsquares is None:
            leastsquares = self._default_options["leastsquares"]
        if leastsquares:
            minimizer = opt.least_squares
            method = kwargs.pop("method", "trf")
            bnds = bnds.T
        else:
            minimizer = opt.minimize
            method = kwargs.pop("method", "slsqp")

        # -----------------------------
        # Fit

        x0 = Quantity([rot0, origin.lon, origin.lat]).to_value(u.deg)

        fit_result: opt.OptimizeResult = minimizer(
            residual, x0=x0, args=(rep, not leastsquares), method=method, bounds=bnds, **kwargs
        )

        # Make fit frame from results
        fit_rot, fit_lon, fit_lat = fit_result.x << u.deg
        fit_rep = USphrRep(lon=fit_lon, lat=fit_lat)
        fit_origin = SkyCoord(
            self.frame.realize_frame(fit_rep, representation_type=self.representation_type),
            copy=False,
        )
        fit_frame = fit_origin.skyoffset_frame(rotation=fit_rot)
        fit_frame.representation_type = self.representation_type
        # there's no data, so setting rep-type here changes it everywhere.

        # TODO!
        # values = dict(rotation=fit_rot, origin=fit_origin)
        # if align_v is None:
        #     align_v = self._default_options["align_v"]
        # if align_v is None and "s" in data.data.differentials:
        #     align_v = True
        # if align_v:
        #     values = self.align_v_positive_lon(values, subsel=...)

        return FrameOptimizeResult(fit_frame, **fit_result)


# -------------------------------------------------------------------


class FrameFitterPlotDescriptor(PlotDescriptorBase["FrameOptimizeResult"]):
    """Plot FrameOptimizeResult."""

    def residual(
        self, stream: "Stream", *, ax: Optional[Axes] = None, format_ax: bool = True
    ) -> Axes:
        """Residual plot of fitting the frame.

        Parameters
        ----------
        stream : `trackstream.stream.base.StreamBase`
        ax : Optional[Axes], optional
            Matplotlib |Axes|, by default None
        format_ax : bool, optional
            Whether to add the axes labels and info, by default True

        Returns
        -------
        |Axes|
        """
        fr, _ax, *_ = self._setup(ax)

        # Residual plot
        rotation_angles: ndarray = linspace(-180, 180, num=1_000, dtype=float)
        r = fr.origin.data
        res = array(
            [
                residual(
                    (float(angle), float(r.lon.deg), float(r.lat.deg)),
                    stream.data_coords.icrs.cartesian,  # TODO! not require ICRS
                    scalar=True,
                )
                for angle in rotation_angles
            ]
        )
        _ax.scatter(rotation_angles, res)

        # Plot the best-fit rotation
        _ax.axvline(fr.rotation.value, c="k", ls="--", label="best-fit rotation")
        # and the next period
        next_period = 180 if (fr.rotation.value - 180) < rotation_angles.min() else -180
        _ax.axvline(fr.rotation.value + next_period, c="k", ls="--", alpha=0.5)

        if format_ax:
            _ax.set_xlabel(r"Rotation angle $\theta$", fontsize=13)
            _ax.set_ylabel(r"Residual / # data pts", fontsize=13)
            _ax.legend()

        return _ax


class FrameOptimizeResult(opt.OptimizeResult, CommonBase):
    """Result of fitting a rotated frame.

    Parameters
    ----------
    origin : `~astropy.coordinates.SkyCoord`
        The location of point on sky about which to rotate.
    rotation : Quantity['angle']
        The rotation about the ``origin``.
    **kwargs : Any
        Fit results. See `~scipy.optimize.OptimizeResult`.
    """

    plot = FrameFitterPlotDescriptor()

    def __init__(self, frame: SkyOffsetFrame, **kwargs: Any) -> None:
        super().__init__(**kwargs)  # setting from OptimizeResult
        CommonBase.__init__(
            self, frame=frame, representation_type=frame.representation_type, differential_type=None
        )

    @property
    def origin(self) -> BaseCoordinateFrame:
        """The location of point on sky."""
        return self._frame.origin

    @property
    def rotation(self) -> Quantity:
        """The rotation about the ``origin``."""
        return self._frame.rotation

    def calculate_residual(self, data: SkyCoord, scalar: bool = False) -> Quantity:
        """Calculate result residual given the fit frame.

        Parameters
        ----------
        data : (N,) `~astropy.coordinates.SkyCoord`
        scalar : bool
            Whether to sum the results.

        Returns
        -------
        `~astropy.units.Quantity`
            Scalar if ``scalar``, else length N.
        """
        ur = data.transform_to(self.frame).represent_as(USphrRep)
        res: Quantity = abs(ur.lat - 0.0 * u.rad)
        return sum(res) if scalar else res

    # ---------------------

    def __repr__(self) -> str:
        """String representation. Adapted from `scipy.optimize.OptimizeResult`."""
        if self.keys():
            header = object.__repr__(self)
            m = max(map(len, list(self.keys()))) + 1  # same-column colon

            contents = [
                (k[1:] if k.startswith("_") else k).rjust(m) + ": " + repr(v)
                for k, v in sorted(self.items())
            ]

            return header + "\n" + "\n".join(contents)

        else:  # if no values
            return self.__class__.__name__ + "()"
