# -*- coding: utf-8 -*-

"""Fit a Rotated ICRS reference frame."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import copy
from types import FunctionType, MappingProxyType, MethodType
from typing import Any, Dict, Optional, Sequence, Tuple, TypedDict, TypeVar, Union

# THIRD PARTY
import astropy.units as u
import numpy as np
import scipy.optimize as opt
from astropy.coordinates import (
    BaseCoordinateFrame,
    CartesianRepresentation,
    SkyCoord,
    SkyOffsetFrame,
    SphericalCosLatDifferential,
    SphericalRepresentation,
    UnitSphericalRepresentation,
)
from astropy.units import Quantity
from astropy.utils.decorators import lazyproperty
from erfa import ufunc as erfa_ufunc
from numpy import ndarray

# LOCAL
from trackstream._type_hints import EllipsisType
from trackstream.utils import reference_to_skyoffset_matrix

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
    data: CartesianRepresentation,
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
    rot_xyz = np.dot(rot_matrix, data.xyz).T

    # cartesian to spherical
    r = np.sqrt(np.sum(np.square(rot_xyz), axis=-1))
    lon, lat = erfa_ufunc.c2s(rot_xyz)

    return r, lon, lat


# -------------------------------------------------------------------


def residual(
    variables: Tuple[float, float, float],
    data: CartesianRepresentation,
    scalar: bool = False,
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
    scalar : bool (optional, keyword-only)
        Whether to sum `res` into a float.
        Note that if `res` is also a float, it is unaffected.
    """
    rotation = variables[0]
    lon = variables[1]
    lat = variables[2]

    # # Cartesian model  # TODO?
    # rot_matrix = reference_to_skyoffset_matrix(lon, lat, rotation)
    # rot_xyz = np.dot(rot_matrix, data.xyz.value).reshape(-1, len(data))

    _, _, phi2 = cartesian_model(data, lon=lon, lat=lat, rotation=rotation)
    # Residual
    res: ndarray = np.abs(phi2.to_value(u.deg) - 0.0) / len(phi2)  # phi2 - 0

    if scalar:
        sres: float = np.sum(res)
        return sres
    return res


#####################################################################


class RotatedFrameFitter(object):
    """Class to Fit Rotated Frames.

    Parameters
    ----------
    data : :class:`~astropy.coordinates.BaseCoordinateFrame`
        In ICRS coordinates.

    origin : :class:`~astropy.coordinates.ICRS`
        The location of point on sky about which to rotate.

    Other Parameters
    ----------------
    rot_lower, rot_upper : |Quantity|, (optional, keyword-only)
        The lower and upper bounds in degrees.
        Default is (-180, 180] degree.
    origin_lim : |Quantity|, (optional, keyword-only)
        The symmetric lower and upper bounds on origin in degrees.
        Default is 0.005 degree.

    fix_origin : bool (optional, keyword-only)
        Whether to fix the origin point. Default is False.
    leastsquares : bool (optional, keyword-only)
        Whether to to use :func:`~scipy.optimize.least_square` or
        :func:`~scipy.optimize.minimize`. Default is False.

    align_v : bool
        Whether to align by the velocity.
    """

    _data: BaseCoordinateFrame
    _origin: BaseCoordinateFrame
    _bounds: ndarray
    _fitter_kwargs: Dict[str, Any]

    def __init__(
        self, data: BaseCoordinateFrame, origin: BaseCoordinateFrame, **kwargs: Any
    ) -> None:
        super().__init__()
        self._data = data
        self._origin = origin

        # Create bounds
        bounds_args = ("rot_lower", "rot_upper", "origin_lim")
        bounds_kwargs = {k: kwargs.pop(k) for k in bounds_args if k in kwargs}
        self.set_bounds(**bounds_kwargs)

        # Process options
        # determine whether velocity exists to break +/- 180 degree
        # degeneracy If it does, call the `align_v` option in `fit_frame`
        align_v = kwargs.pop("align_v", None)
        if align_v and "s" not in self.data.data.differentials:
            raise ValueError
        if align_v is None and "s" in self.data.data.differentials:
            align_v = True

        self._default_options = _RotatedFrameFitterOptions(
            fix_origin=kwargs.pop("fix_origin", False),
            leastsquares=kwargs.pop("leastsquares", False),
            align_v=align_v,
        )

        # Minimizer kwargs are the leftovers
        self._fitter_kwargs = kwargs

    @property
    def data(self) -> BaseCoordinateFrame:
        return self._data

    @property
    def origin(self) -> BaseCoordinateFrame:
        return self._origin

    @property
    def bounds(self) -> ndarray:
        return self._bounds

    @property
    def fitter_kwargs(self) -> Dict[str, Any]:
        return self._fitter_kwargs

    @property
    def default_fit_options(self) -> MappingProxyType:
        """The default fit options, including from initialization."""
        return MappingProxyType(dict(**self._default_options, **self.fitter_kwargs))

    #######################################################

    # @u.quantity_input(rot_lower=u.deg, rot_upper=u.deg, origin_lim=u.deg)
    def set_bounds(
        self,
        rot_lower: Quantity = -180.0 * u.deg,
        rot_upper: Quantity = 180.0 * u.deg,
        origin_lim: Quantity = 0.005 * u.deg,
    ) -> None:
        """Make bounds on Rotation parameter.

        Parameters
        ----------
        rot_lower, rot_upper : |Quantity|, optional
            The lower and upper bounds in degrees.
        origin_lim : |Quantity|, optional
            The symmetric lower and upper bounds on origin in degrees.
        """
        origin = self.origin.represent_as(UnitSphericalRepresentation)

        rotation_bounds = (rot_lower.to_value(u.deg), rot_upper.to_value(u.deg))
        # longitude bounds (ra in ICRS).
        lon_bounds = (origin.lon + (-1, 1) * origin_lim).to_value(u.deg)
        # latitude bounds (dec in ICRS).
        lat_bounds = (origin.lat + (-1, 1) * origin_lim).to_value(u.deg)

        # stack bounds so rows are bounds.
        bounds = np.c_[rotation_bounds, lon_bounds, lat_bounds].T

        self._bounds = bounds

    def align_v_positive_lon(
        self,
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
        values = copy.deepcopy(fit_values)  # copy for safety
        rotation = values["rotation"]

        # make frame
        frame = SkyOffsetFrame(**values)  # make frame
        frame.differential_type = SphericalCosLatDifferential

        rot_data = self.data.transform_to(frame)
        # rot_datarot_data.represent_as(coord.SphericalRepresentation)

        # # all this to get the rotated velocity
        # # TODO faster!
        # rot_matrix = reference_to_skyoffset_matrix(
        #     lon=origin.lon, lat=origin.lat, rotation=rotation
        # )
        # rot_data = data.transform(rot_matrix).represent_as(
        #     coord.SphericalRepresentation,
        #     differential_class=coord.SphericalCosLatDifferential,
        # )
        # rot_vel = rot_data.differentials["s"]

        # get average velocity to determine whether need to rotate.
        # TODO determine whether
        avg = np.median(rot_data.pm_lon_coslat[subsel])

        if avg < 0:  # need to flip
            rotation = rotation + 180 * u.deg

        return values

    #######################################################
    # Fitting

    def residual(self, rotation: float, *, scalar: bool = False) -> ndarray:
        r"""How close phi2, the rotated latitude (dec), is to flat.

        Parameters
        ----------
        rotation : float
            The final rotation of the frame about the ``origin``. The sign of
            the rotation is the left-hand rule.  That is, an object at a
            particular position angle in the un-rotated system will be sent to
            the positive latitude (z) direction in the final frame.
            In degrees.

        Returns
        -------
        res : float or Sequence
            :math:`\rm{lat} - 0`.
            If `scalar` is True, then sum array_like to return float.

        Other Parameters
        ----------------
        scalar : bool (optional, keyword-only)
            Whether to sum `res` into a float.
            Note that if `res` is also a float, it is unaffected.
        """
        origin = self.origin.represent_as(SphericalRepresentation)
        variables = (rotation, origin.lon.to_value(u.deg), origin.lat.to_value(u.deg))
        rsdl = np.asanyarray(residual(variables, self.data.cartesian, scalar=scalar))
        return rsdl

    def _fit_representation_scipy(
        self,
        data: CartesianRepresentation,
        x0: Sequence[float],
        *,
        bounds: ndarray,
        fix_origin: bool,
        use_leastsquares: bool,
        **kw: Any,
    ) -> Tuple[opt.OptimizeResult, Quantity[u.deg]]:
        if fix_origin:
            bounds[1, :] = np.average(bounds[1, :])
            bounds[2, :] = np.average(bounds[2, :])
            raise NotImplementedError("TODO")

        if use_leastsquares:
            method = kw.pop("method", "trf")
            result = opt.least_squares(
                residual, x0=x0, args=(data, False), method=method, bounds=bounds.T, **kw
            )
        else:
            method = kw.pop("method", "slsqp")
            result = opt.minimize(
                residual, x0=x0, args=(data, True), method=method, bounds=bounds, **kw
            )

        values = result.x << u.deg

        return result, values

    def fit(
        self,
        rot0: Optional[Quantity] = None,
        bounds: Optional[ndarray] = None,
        *,
        fix_origin: Optional[bool] = None,
        leastsquares: Optional[bool] = None,
        align_v: Optional[bool] = None,
        **kwargs: Any,
    ) -> FitResult:
        """Find Best-Fit Rotated Frame.

        Parameters
        ----------
        rot0 : |Quantity|, optional
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
        fix_origin : bool (optional, keyword-only)
            Whether to fix the origin.
        leastsquares : bool (optional, keyword-only)
            Whether to to use :func:`~scipy.optimize.least_square` or
            :func:`~scipy.optimize.minimize` (default).
        align_v : bool (optional, keyword-only)
            Whether to align velocity to be in positive direction
        fit_kwargs:
            Into whatever minimization package / function is used.
        """
        # -----------------------------
        # Prepare, using defaults for arguments not provided.

        if rot0 is None:
            rot0 = self.fitter_kwargs.get("rot0", None)
            if rot0 is None:
                raise ValueError("no prespecified `rot0`; Need to provide one.")

        bounds = self.bounds if bounds is None else bounds
        if fix_origin is None:
            fix_origin = self._default_options["fix_origin"]
        if leastsquares is None:
            leastsquares = self._default_options["leastsquares"]
        if align_v is None:
            align_v = self._default_options["align_v"]

        # kwargs, preferring newer
        kwargs = {**self.fitter_kwargs, **kwargs}

        # -----------------------------

        # Origin
        # We work with a SphericalRepresentation, but
        origin_frame = self.origin.__class__
        origin = self.origin.represent_as(SphericalRepresentation)

        x0 = Quantity([rot0, origin.lon, origin.lat]).to_value(u.deg)
        subsel = kwargs.pop("subsel", Ellipsis)

        # Fit with scipy
        leastsquares = bool(kwargs.pop("use_leastsquares", False))
        fit_result, values = self._fit_representation_scipy(
            self.data.cartesian,
            x0=x0,
            bounds=bounds,
            fix_origin=fix_origin,
            use_leastsquares=leastsquares,
            **kwargs,
        )

        # -----------------------------

        best_rot = values[0]
        best_origin = UnitSphericalRepresentation(
            lon=values[1],
            lat=values[2],  # TODO re-add distance
        )
        best_origin = origin_frame(best_origin)

        values = dict(rotation=best_rot, origin=best_origin)
        if align_v:
            values = self.align_v_positive_lon(values, subsel=subsel)

        return FitResult(self.data, fitresult=fit_result, **values)


# -------------------------------------------------------------------


class FitResult:
    """Result of Fit.

    Parameters
    ----------
    data : SkyCoord
    origin : SkyCoord
    rotation: Quantity['angle']
    fitresult : Any, optional

    Attributes
    ----------
    data : |Frame|
        Transformed to |SkyOffsetFrame|
    fitresult : Any, optional
    fit_values : MappingProxy
        Has keys "rotation" and "origin".
    frame
    residual
    residual_scalar

    Methods
    -------
    plot_data
    plot_on_residual
    """

    def __init__(
        self, data: SkyCoord, origin: SkyCoord, rotation: Quantity, fitresult: Optional[Any] = None
    ) -> None:
        self._origin = origin
        self._rotation = rotation
        self.fitresult = fitresult

        self.data = data.transform_to(self.frame)

    @property
    def origin(self) -> SkyCoord:
        return self._origin

    @property
    def rotation(self) -> Quantity:
        return self._rotation

    @lazyproperty
    def fit_values(self) -> MappingProxyType:
        return MappingProxyType(dict(origin=self.origin, rotation=self.rotation))

    @lazyproperty
    def frame(self) -> BaseCoordinateFrame:
        """`~astropy.coordinates.SkyOffsetFrame`."""
        # make frame  # TODO ensure same as `make_frame`
        frame = SkyOffsetFrame(**self.fit_values)
        frame.differential_type = SphericalCosLatDifferential
        return frame

    @lazyproperty
    def residual(self) -> Quantity:
        """Fit result residual."""
        lat: Quantity = self.data.lat
        res = np.abs(lat - 0.0 * u.rad)
        return res

    @property
    def residual_scalar(self) -> Quantity:
        res: Quantity = np.sum(self.residual)
        return res

    @lazyproperty
    def lon_order(self) -> ndarray:
        """Order data by longitude.

        Returns
        -------
        order : ndarray

        """
        orderer = np.argsort(self.data.lon)

        return orderer

    # ---------------------

    def __repr__(self) -> str:
        return f"FitResult({self.fit_values})"
