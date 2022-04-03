# -*- coding: utf-8 -*-

"""Fit a Rotated ICRS reference frame."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import copy
import functools
from types import MappingProxyType, MethodType, FunctionType
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
    TypedDict,
    Callable,
    cast,
)

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import scipy.optimize as opt
from astropy.units import Quantity, PhysicalType
from astropy.coordinates import (
    SkyOffsetFrame,
    SphericalCosLatDifferential,
    CartesianRepresentation,
    BaseCoordinateFrame,
    ICRS,
    UnitSphericalRepresentation,
    SphericalRepresentation,
    SkyCoord,
)
from astropy.utils.decorators import lazyproperty
from erfa import ufunc as erfa_ufunc
from matplotlib.pyplot import Figure
from matplotlib.axes._subplots import AxesSubplot
from numpy import ndarray

# LOCAL
from trackstream.config import conf
from trackstream.setup_package import HAS_LMFIT
from trackstream.utils import reference_to_skyoffset_matrix
from trackstream._type_hints import EllipsisType

if HAS_LMFIT:
    # THIRD PARTY
    import lmfit as lf


__all__ = ["RotatedFrameFitter", "residual"]


##############################################################################
# PARAMETERS

FT = TypeVar("FT", MethodType, FunctionType)


class _RotatedFrameFitterOptions(TypedDict):
    fix_origin: bool
    use_lmfit: Optional[bool]
    leastsquares: bool
    align_v: bool


##############################################################################
# CODE
##############################################################################


@overload
def scipy_residual_to_lmfit(function: None, /, *, param_order: List[str]) -> functools.partial:
    ...


@overload
def scipy_residual_to_lmfit(function: FT, /, *, param_order: List[str]) -> FT:  # noqa: F811
    ...


def scipy_residual_to_lmfit(
    function: Optional[FT] = None, /, *, param_order: List[str]
) -> Union[functools.partial, FT]:  # noqa: F811
    """Decorator to make scipy residual functions compatible with lmfit.

    Parameters
    ----------
    function : callable, positional-only
        The residual function.
    param_order : list of strs
        The variable order used by lmfit.
        Strings are the names of the lmfit parameters.
        Must be in the same order as the scipy residual function.

    Returns
    -------
    function : callable
        The same as ``function``.
    """
    # allow for @-syntax
    if function is None:
        return functools.partial(scipy_residual_to_lmfit, param_order=param_order)

    def lmfit(params: Mapping[str, Any], *args: Any, **kwargs: Any) -> Sequence:
        """:mod:`lmfit` version of function.

        Parameters
        ----------
        params : `~lmfit.Parameters`
        *args, **kwargs : Any
        """
        variables: List[Any] = [params[n].value for n in param_order]
        res: Sequence = cast(FT, function)(variables, *args, **kwargs)
        return res

    # attach lmfit version to original function
    function.lmfit = lmfit  # type: ignore

    return function


# -------------------------------------------------------------------


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


@scipy_residual_to_lmfit(None, param_order=["rotation", "lon", "lat"])
def residual(
    variables: Sequence[float],
    data: CartesianRepresentation,
    scalar: bool = False,
) -> Union[float, ndarray]:
    r"""How close phi2, the rotated latitude (dec), is to flat.

    Parameters
    ----------
    variables : Sequence[float]
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

    # Cartesian model
    rot_matrix = reference_to_skyoffset_matrix(lon, lat, rotation)
    rot_xyz = np.dot(rot_matrix, data.xyz.value).reshape(-1, len(data))

    _, _, phi2 = cartesian_model(data, lon=variables[1], lat=variables[2], rotation=variables[0])
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
    use_lmfit : bool or None, (optional, keyword-only)
        Whether to use ``lmfit`` package.
        None (default) falls back to config file.
    leastsquares : bool (optional, keyword-only)
        If `use_lmfit` is False, whether to to use
        :func:`~scipy.optimize.least_square` or
        :func:`~scipy.optimize.minimize`
        Default is False

    align_v : bool
        Whether to align by the velocity.
    """

    data: BaseCoordinateFrame
    origin: ICRS
    bounds: ndarray
    fitter_kwargs: Dict[str, Any]
    # _default_options = _RotatedFrameFitterOptions

    def __init__(self, data: BaseCoordinateFrame, origin: ICRS, **kwargs: Any) -> None:
        super().__init__()
        self.data = data
        self.origin = origin

        # -------------
        # create bounds

        bounds_args = ("rot_lower", "rot_upper", "origin_lim")
        bounds_kwargs = {k: kwargs.pop(k) for k in bounds_args if k in kwargs}
        self.set_bounds(**bounds_kwargs)

        # -------------
        # process options

        # determine whether velocity exists to break +/- 180 degree
        # degeneracy If it does, call the `align_v` option in `fit_frame`
        align_v = kwargs.pop("align_v", None)
        if align_v and "s" not in self.data.data.differentials:
            raise ValueError
        if align_v is None and "s" in self.data.data.differentials:
            align_v = True

        self._default_options = _RotatedFrameFitterOptions(
            fix_origin=kwargs.pop("fix_origin", False),
            use_lmfit=kwargs.pop("use_lmfit", None),
            leastsquares=kwargs.pop("leastsquares", False),
            align_v=align_v,
        )

        # Minimizer kwargs are the leftovers
        self.fitter_kwargs = kwargs

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

        self.bounds = bounds

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
        variables = (rotation, self.origin.ra.to_value(u.deg), self.origin.dec.to_value(u.deg))
        rsdl: ndarray = residual(variables, self.data.cartesian, scalar=scalar)
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
                residual,
                x0=x0,
                args=(data, False),
                method=method,
                bounds=bounds.T,
                **kw,
            )

        else:
            method = kw.pop("method", "slsqp")
            result = opt.minimize(
                residual,
                x0=x0,
                args=(data, True),
                method=method,
                bounds=bounds,
                **kw,
            )

        values = result.x << u.deg

        return result, values

    def _fit_representation_lmfit(
        self,
        data: CartesianRepresentation,
        x0: Sequence[float],
        *,
        bounds: ndarray,
        fix_origin: bool,
        **kw: Any,
    ) -> Tuple[object, Quantity[u.deg]]:
        if np.shape(bounds) == (2,):
            rot_bnd = lon_bnd = lat_bnd = bounds
        elif np.shape(bounds) == (3, 2):
            rot_bnd, lon_bnd, lat_bnd = bounds

        params = lf.Parameters()
        params.add_many(
            ("rotation", x0[0], True, rot_bnd[0], rot_bnd[1]),
            ("lon", x0[1], not fix_origin, lon_bnd[0], lon_bnd[1]),
            ("lat", x0[2], not fix_origin, lat_bnd[0], lat_bnd[1]),
        )

        method = kw.pop("method", "powell")

        res = lf.minimize(
            residual.lmfit,
            params,
            kws=dict(data=data, scalar=False),
            method=method,
            calc_covar=True,
            **kw,
        )

        values = np.array(tuple(res.params.valuesdict().values())) * u.deg

        return res, values

    # @u.quantity_input(rot0=u.deg)
    def fit(
        self,
        rot0: Optional[Quantity] = None,
        bounds: Optional[ndarray] = None,
        *,
        fix_origin: Optional[bool] = None,
        use_lmfit: Optional[bool] = None,
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
        use_lmfit : bool (optional, keyword-only)
            Whether to use ``lmfit`` package
        leastsquares : bool (optional, keyword-only)
            If `use_lmfit` is False, whether to to use
            :func:`~scipy.optimize.least_square` or
            :func:`~scipy.optimize.minimize` (default)
        align_v : bool (optional, keyword-only)
            Whether to align velocity to be in positive direction
        fit_kwargs:
            Into whatever minimization package / function is used.

        Raises
        ------
        ImportError
            If ``use_lmfit`` and :mod:`lmfit` is not installed.
        """
        # -----------------------------
        # Prepare, using defaults for arguments not provided.

        if rot0 is None:
            rot0 = self.fitter_kwargs.get("rot0", None)
            if rot0 is None:
                raise ValueError("no prespecified `rot0`; Need to provide one.")

        if bounds is None:
            bounds = self.bounds
        if fix_origin is None:
            fix_origin = self._default_options["fix_origin"]
        if use_lmfit is None:
            use_lmfit = self._default_options["use_lmfit"]
            if use_lmfit is None:  # still None
                use_lmfit = conf.use_lmfit
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

        if use_lmfit:  # lmfit
            if not HAS_LMFIT:
                raise ImportError("`lmfit` package not available.")

            fit_result, values = self._fit_representation_lmfit(
                self.data.cartesian,
                x0=x0,
                bounds=bounds,
                fix_origin=fix_origin,
                **kwargs,
            )

        else:  # scipy
            use_leastsquares = kwargs.pop("use_leastsquares", None)
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

    #######################################################
    # Plot

    #     def plot_data(self):
    #         # THIRD PARTY
    #         import matplotlib.pyplot as plt
    #
    #         plt.scatter(self.data.ra, self.data.dec)
    #         # plt.ylim(-90, 90)
    #
    #         # return fig

    def plot_residual(
        self,
        fitresult: Optional[FitResult] = None,
        num_rots: int = 3600,
        scalar: bool = True,
    ) -> Tuple[Figure, AxesSubplot]:
        """Plot Residual as a function of rotation angle."""
        # LOCAL
        from .plot import plot_rotation_frame_residual

        fig, ax = plot_rotation_frame_residual(
            self.data,
            self.origin,
            num_rots=num_rots,
            scalar=scalar,
        )

        if fitresult is not None:
            fitresult.plot_on_residual(ax, scalar=scalar)

        return fig, ax


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

    # ---------------------

    def plot_data(self, ax: AxesSubplot) -> None:
        ax.scatter(self.data.lon, self.data.lat)
        ax.set_ylim(-90, 90)

        # return fig

    def plot_on_residual(self, ax: AxesSubplot, scalar: bool = True) -> None:
        if scalar:
            theta = self.fit_values["rotation"]
            # plt.axvline(theta)
            ax.scatter(theta.to_value(u.deg), self.residual_scalar, c="r")

        else:
            raise NotImplementedError
