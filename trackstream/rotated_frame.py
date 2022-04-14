# -*- coding: utf-8 -*-

"""Fit a Rotated ICRS reference frame."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import copy
from types import FunctionType, MappingProxyType, MethodType
from typing import Any, Dict, Optional, Sequence, Tuple, Type, TypedDict, TypeVar, Union

# THIRD PARTY
import astropy.units as u
import numpy as np
import scipy.optimize as opt
from astropy.coordinates import BaseCoordinateFrame, BaseRepresentation, CartesianRepresentation
from astropy.coordinates import SkyCoord, SkyOffsetFrame, SphericalCosLatDifferential
from astropy.coordinates import UnitSphericalRepresentation
from astropy.units import Quantity
from erfa import ufunc as erfa_ufunc
from numpy import ndarray

# LOCAL
from trackstream._type_hints import EllipsisType, FrameLikeType
from trackstream.base import CommonBase
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
    rot_xyz: Quantity = np.dot(rot_matrix, data.xyz).T

    # cartesian to spherical
    r: Quantity = np.sqrt(np.sum(np.square(rot_xyz), axis=-1))
    _lon, _lat = erfa_ufunc.c2s(rot_xyz)

    return r, _lon, _lat


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
    rotation, lon, lat = variables

    _, _, phi2 = cartesian_model(data, lon=lon, lat=lat, rotation=rotation)
    # Residual
    res: ndarray = np.abs(phi2.to_value(u.deg) - 0.0) / len(phi2)  # phi2 - 0

    if scalar:
        sres: float = np.sum(res)
        return sres
    return res


#####################################################################


class RotatedFrameFitter(CommonBase):
    """Class to Fit Rotated Frames.

    The fitting is always on-sky.

    Parameters
    ----------
    origin : :class:`~astropy.coordinates.ICRS`
        The location of point on sky about which to rotate.

    frame : `~astropy.coordinates.BaseCoordinateFrame` or None, optional keyword-only
        The frame. If `None` (default) uses the frame of `origin`.
    representation_type : `astropy.coordinates.BaseRepresentation` or None, optional keyword-only
        The representation type for the `frame`. If `None` (default) uses
        the current representation type of the `frame`.
        The fitting happens in `~astropy.coordinates.UnitSphericalRepresentation`
        but will be returned in this `representation_type`.

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

    _origin: SkyCoord
    _bounds: ndarray
    _fitter_kwargs: Dict[str, Any]

    def __init__(
        self,
        origin: SkyCoord,
        *,
        frame: Optional[FrameLikeType] = None,
        representation_type: Optional[Type[BaseRepresentation]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            frame=origin if frame is None else frame,
            representation_type=representation_type,
        )
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
        self._fitter_kwargs = kwargs

    @property
    def origin(self) -> SkyCoord:
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
        values = copy.deepcopy(fit_values)  # copy for safety
        rotation = values["rotation"]

        # make frame
        frame = SkyOffsetFrame(**values)  # make frame
        frame.differential_type = SphericalCosLatDifferential

        rot_data = data.transform_to(frame)

        # get average velocity to determine whether need to rotate.
        # TODO determine whether
        avg = np.median(rot_data.pm_lon_coslat[subsel])

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
        align_v: Optional[bool] = None,
        **kwargs: Any,
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
        # Put data in right coordinates & cartesian representation
        # but routed through UnitSphericalRepresentation
        data = data.transform_to(self.frame)
        data_r: CartesianRepresentation
        data_r = data.represent_as(UnitSphericalRepresentation).represent_as(
            CartesianRepresentation,
        )

        # Put origin in right representation type
        origin: UnitSphericalRepresentation
        origin = self.origin.represent_as(UnitSphericalRepresentation)

        # -----------------------------
        # Prepare, using defaults for arguments not provided.

        # kwargs, preferring newer
        kwargs = {**self.fitter_kwargs, **kwargs}

        if rot0 is None:
            rot0 = self.fitter_kwargs.get("rot0", None)
            if rot0 is None:
                raise ValueError("no prespecified `rot0`; need to provide one.")

        _bounds = self.bounds if bounds is None else bounds

        # Origin
        if fix_origin is None:
            fix_origin = self._default_options["fix_origin"]
        if fix_origin:
            _bounds[1, :] = np.average(_bounds[1, :])
            _bounds[2, :] = np.average(_bounds[2, :])
            raise NotImplementedError("TODO")

        # Process fit options
        if leastsquares is None:
            leastsquares = self._default_options["leastsquares"]
        if leastsquares:
            minimizer = opt.least_squares
            method = kwargs.pop("method", "trf")
            _bounds = _bounds.T
        else:
            minimizer = opt.minimize
            method = kwargs.pop("method", "slsqp")
            _bounds = _bounds

        # -----------------------------
        # Fit

        x0 = Quantity([rot0, origin.lon, origin.lat]).to_value(u.deg)

        fit_result: opt.OptimizeResult
        fit_result = minimizer(
            residual,
            x0=x0,
            args=(data_r, not leastsquares),
            method=method,
            bounds=_bounds,
            **kwargs,
        )
        # Extract result
        fit_rot, fit_lon, fit_lat = fit_result.x << u.deg

        # -----------------------------

        # TODO? re-add distance
        fit_origin_r = UnitSphericalRepresentation(lon=fit_lon, lat=fit_lat)
        fit_origin = SkyCoord(self.frame.realize_frame(fit_origin_r), copy=False)
        fit_frame = fit_origin.skyoffset_frame(rotation=fit_rot)
        fit_frame.representation_type = self.representation_type

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


class FrameOptimizeResult(opt.OptimizeResult, CommonBase):
    """Result of Fit.

    Parameters
    ----------
    data : SkyCoord
    origin : SkyCoord
    rotation: Quantity['angle']
    fitresult : Any, optional
    """

    def __init__(
        self,
        frame: SkyOffsetFrame,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)  # setting from OptimizeResult
        CommonBase.__init__(self, frame=frame, representation_type=frame.representation_type)

    @property
    def origin(self) -> BaseCoordinateFrame:
        return self._frame.origin

    @property
    def rotation(self) -> Quantity:
        return self._frame.rotation

    def calculate_residual(self, data: SkyCoord, scalar: bool = False) -> Quantity:
        """Fit result residual."""
        ur = data.transform_to(self.frame).represent_as(UnitSphericalRepresentation)
        lat: Quantity = ur.lat
        res: Quantity = np.abs(lat - 0.0 * u.rad)
        if scalar:
            res = np.sum(res)
        return res

    # ---------------------

    def __repr__(self) -> str:
        if self.keys():
            header = object.__repr__(self)
            m = max(map(len, list(self.keys()))) + 1
            return (
                header
                + "\n"
                + "\n".join(
                    [
                        (k[1:] if k.startswith("_") else k).rjust(m) + ": " + repr(v)
                        for k, v in sorted(self.items())
                    ],
                )
            )

        else:  # if no values
            return self.__class__.__name__ + "()"
