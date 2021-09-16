# -*- coding: utf-8 -*-

"""Fit a Rotated ICRS reference frame.

.. todo::

    Use an Astropy Model instead.


"""


__all__ = [
    "RotatedFrameFitter",
    "cartesian_model",
    "residual",
    "align_v_positive_lon",
]


##############################################################################
# IMPORTS

# STDLIB
import copy
import functools
import typing as T
from types import MappingProxyType

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import scipy.optimize as opt
from astropy.utils.decorators import lazyproperty

# LOCAL
from .utils import cartesian_to_spherical, reference_to_skyoffset_matrix
from trackstream._type_hints import QuantityType
from trackstream.config import conf
from trackstream.setup_package import HAS_LMFIT

if HAS_LMFIT:
    # THIRD PARTY
    import lmfit as lf


##############################################################################
# CODE
##############################################################################


def scipy_residual_to_lmfit(function=None, param_order=None):
    """Decorator to make scipy residual functions compatible with lmfit.

    Parameters
    ----------
    function : callable
        The residual function.
    param_order : list of strs
        The variable order used by lmfit.
        Strings are the names of the lmfit parameters.
        Must be in the same order as the scipy residual function.
    """
    if param_order is None:
        raise ValueError("`param_order` cannot be None")

    if function is None:
        return functools.partial(scipy_residual_to_lmfit, param_order=param_order)

    def lmfit(params, *args: T.Any, **kwargs: T.Any) -> T.Sequence:
        """`lmfit` version of function."""
        variables = [params[n].value for n in param_order]
        return function(variables, *args, **kwargs)

    # /def

    function.lmfit = lmfit

    return function


# /def


def cartesian_model(
    data: coord.CartesianRepresentation,
    *,
    lon: T.Union[QuantityType, float],
    lat: T.Union[QuantityType, float],
    rotation: T.Union[QuantityType, float],
    deg: bool = True,
) -> T.Tuple:
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

    Other Parameters
    ----------------
    deg : bool
        whether to return `lat` and `lon` as degrees
        (default True) or radians.

    """
    rot_matrix = reference_to_skyoffset_matrix(lon, lat, rotation)
    rot_xyz = np.dot(rot_matrix, data.xyz.value).reshape(-1, len(data))

    r, lat, lon = cartesian_to_spherical(*rot_xyz, deg=deg)

    return r, lon, lat


# /def


# -------------------------------------------------------------------


@scipy_residual_to_lmfit(param_order=["rotation", "lon", "lat"])
def residual(
    variables: T.Sequence,
    data: coord.CartesianRepresentation,
    scalar: bool = False,
) -> T.Union[float, T.Sequence]:
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
    res : float or Sequence
        :math:`\rm{lat} - 0`

        If `scalar` is True, then sum array_like to return float.

    Other Parameters
    ----------------
    scalar : bool
        Whether to sum `res` into a float.
        Note that if `res` is also a float, it is unaffected.

    """
    rotation = variables[0]
    lon = variables[1]
    lat = variables[2]

    r, lon, lat = cartesian_model(
        data,
        lon=lon,
        lat=lat,
        rotation=rotation,
        deg=True,
    )

    res = np.abs(lat - 0.0)  # phi2 - 0

    if scalar:
        return np.sum(res)
    return res


# /def

# -------------------------------------------------------------------


def _make_frame(**fit_values: T.Dict[str, T.Any]):
    """Thin Wrapper for `~astropy.coordinates.SkyOffsetFrame`.

    Parameters
    ----------
    fit_values : Dict[str, Any]
        Results of minimization.
        See `~trackstream.preprocess.fit_rotated_frame.minimize`

    Returns
    -------
    frame : SkyOffsetFrame

    """
    frame = coord.SkyOffsetFrame(**fit_values)  # make frame
    frame.differential_type = coord.SphericalCosLatDifferential

    return frame


# /def


# -------------------------------------------------------------------


def align_v_positive_lon(
    data: coord.BaseCoordinateFrame,
    fit_values: T.Dict[str, T.Any],
    subsel: T.Union[type(Ellipsis), T.Sequence, slice] = Ellipsis,
):
    """Align the velocity along the positive Longitudinal direction.

    Parameters
    ----------
    data : Coordinate
        Must have differentials
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

    rot_data = data.transform_to(_make_frame(**values))
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


# /def


#####################################################################


class RotatedFrameFitter(object):
    """Class to Fit Rotated Frames.

    .. todo::

        include errors.

    Parameters
    ----------
    data : :class:`~astropy.coordinates.BaseCoordinateFrame`
        In ICRS coordinates.

    origin : :class:`~astropy.coordinates.ICRS`
        location of point on sky about which to rotate.

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

    Raises
    ------
    ValueError
        If

    """

    def __init__(self, data: coord.BaseCoordinateFrame, origin: coord.ICRS, **kwargs):
        super().__init__()
        self.data = data
        self.origin = origin

        # -------------
        # create bounds

        bounds_args = {
            k: kwargs.pop(k) for k in ("rot_lower", "rot_upper", "origin_lim") if k in kwargs
        }
        self.set_bounds(**bounds_args)

        # -------------
        # process options

        self._default_options = dict(
            fix_origin=kwargs.pop("fix_origin", False),
            use_lmfit=kwargs.pop("use_lmfit", None),
            leastsquares=kwargs.pop("leastsquares", False),
        )
        # determine whether velocity exists to break +/- 180 degree
        # degeneracy If it does, call the `align_v` option in `fit_frame`
        align_v = kwargs.pop("align_v", None)
        if align_v and "s" not in self.data.data.differentials:
            raise ValueError
        if align_v is None and "s" in self.data.data.differentials:
            align_v = True

        self._default_options["align_v"] = align_v

        # Minimizer kwargs are the leftovers
        self.fitter_kwargs = kwargs

    # /def

    @property
    def default_fit_options(self):
        return MappingProxyType(
            dict(**self._default_options, **self.fitter_kwargs),
        )

    #######################################################

    # @u.quantity_input(rot_lower=u.deg, rot_upper=u.deg, origin_lim=u.deg)
    def set_bounds(
        self,
        rot_lower: u.Quantity = -180.0 * u.deg,
        rot_upper: u.Quantity = 180.0 * u.deg,
        origin_lim: u.Quantity = 0.005 * u.deg,
    ) -> T.Tuple[float, float]:
        """Make bounds on Rotation parameter.

        Parameters
        ----------
        rot_lower, rot_upper : |Quantity|, optional
            The lower and upper bounds in degrees.
        origin_lim : |Quantity|, optional
            The symmetric lower and upper bounds on origin in degrees.

        """
        origin = self.origin.data.represent_as(
            coord.UnitSphericalRepresentation,
        )

        rotation_bounds = (
            rot_lower.to_value(u.deg),
            rot_upper.to_value(u.deg),
        )
        # longitude bounds (ra in ICRS).
        lon_bounds = (origin.lon + (-1, 1) * origin_lim).to_value(u.deg)
        # latitude bounds (dec in ICRS).
        lat_bounds = (origin.lat + (-1, 1) * origin_lim).to_value(u.deg)

        # stack bounds so rows are bounds.
        bounds = np.c_[rotation_bounds, lon_bounds, lat_bounds].T

        self.bounds = bounds

    # /def

    #######################################################
    # Fitting

    def _fit_representation_scipy(
        self,
        data: coord.CartesianRepresentation,
        x0: T.Sequence[float],
        *,
        bounds: np.ndarray,
        fix_origin: bool,
        use_leastsquares: bool,
        **kw,
    ):
        if fix_origin:
            bounds[1, :] = np.average(bounds[1, :])
            bounds[2, :] = np.average(bounds[2, :])

            raise NotImplementedError("TODO")

        if use_leastsquares:
            method = kw.pop("method", "trf")
            res = opt.least_squares(
                residual,
                x0=x0,
                args=(data, False),
                method=method,
                bounds=bounds.T,
                **kw,
            )

        else:
            method = kw.pop("method", "slsqp")
            res = opt.minimize(
                residual,
                x0=x0,
                args=(data, True),
                method=method,
                bounds=bounds,
                **kw,
            )

        values = res.x * u.deg

        return res, values

    # /def

    def _fit_representation_lmfit(
        self,
        data: coord.CartesianRepresentation,
        x0: T.Sequence[float],
        *,
        bounds: np.ndarray,
        fix_origin: bool,
        **kw,
    ):
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

    # /def

    # @u.quantity_input(rot0=u.deg)
    def fit(
        self,
        rot0: T.Optional[u.Quantity] = None,
        bounds: T.Optional[T.Sequence] = None,
        *,
        fix_origin: T.Optional[bool] = None,
        use_lmfit: T.Optional[bool] = None,
        leastsquares: T.Optional[bool] = None,
        align_v: T.Optional[bool] = None,
        **kwargs,
    ):
        """Find Best-Fit Rotated Frame.

        Parameters
        ----------
        rot0 : |Quantity|
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
        Dict[str, Any]
            Has fields "rotation" and "origin".

        Other Parameters
        ----------------
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
        ValueError
            If `use_lmfit` and lmfit is not installed.

        """
        # -----------------------------
        # Prepare

        if rot0 is None:
            rot0 = self.fitter_kwargs.get("rot0", None)
            if rot0 is None:
                raise ValueError(
                    "No prespecified `rot0`; " "Need to provide one.",
                )

        if bounds is None:
            bounds = self.bounds

        if fix_origin is None:
            fix_origin = self._default_options["fix_origin"]
        if use_lmfit is None:
            fix_origin = self._default_options["use_lmfit"]
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
        origin = self.origin.represent_as(coord.SphericalRepresentation)

        x0 = u.Quantity([rot0, origin.lon, origin.lat]).to_value(u.deg)
        subsel = kwargs.pop("subsel", Ellipsis)

        if use_lmfit:  # lmfit
            if not HAS_LMFIT:
                raise ValueError("`lmfit` package not available.")

            fit_result, values = self._fit_representation_lmfit(
                self.data.cartesian,
                x0=x0,
                bounds=bounds,
                fix_origin=fix_origin,
                **kwargs,
            )

        else:  # scipy
            fit_result, values = self._fit_representation_scipy(
                self.data.cartesian,
                x0=x0,
                bounds=bounds,
                fix_origin=fix_origin,
                use_leastsquares=leastsquares,
                **kwargs,
            )

        # /def

        # -----------------------------

        best_rot = values[0]
        best_origin = coord.UnitSphericalRepresentation(
            lon=values[1],
            lat=values[2],  # TODO re-add distance
        )
        best_origin = origin_frame(best_origin)

        values = dict(rotation=best_rot, origin=best_origin)
        if align_v:
            values = align_v_positive_lon(self.data, values, subsel=subsel)

        return FitResult(self.data, fitresult=fit_result, **values)

    # /def

    #######################################################

    def residual(self, rotation, scalar: bool = False):
        """Residual function."""
        variables = (
            rotation,
            self.origin.ra.to_value(u.deg),
            self.origin.dec.to_value(u.deg),
        )
        return residual(variables, self.data, scalar=scalar)

    # /def

    #######################################################
    # Plot

    def plot_data(self):
        # THIRD PARTY
        import matplotlib.pyplot as plt

        plt.scatter(self.data.ra, self.data.dec)
        # plt.ylim(-90, 90)

        # return fig

    # /def

    def plot_residual(
        self,
        fitresult=None,
        num_rots: int = 3600,
        scalar: bool = True,
    ):
        """Plot Residual as a function of rotation angle."""
        # LOCAL
        from .plot import plot_rotation_frame_residual

        fig = plot_rotation_frame_residual(
            self.data,
            self.origin,
            num_rots=num_rots,
            scalar=scalar,
        )

        if fitresult is not None:
            fitresult.plot_on_residual(scalar=scalar)

        return fig

    # /def


# /class

# -------------------------------------------------------------------


class FitResult:
    """Result of Fit.

    Parameters
    ----------
    data : |Frame|
        In ICRS coordinates.
    fit_values : Dict[str, Any]
        Has keys "rotation" and "origin".
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

    def __init__(self, data, origin, rotation, fitresult=None):
        self._origin = origin
        self._rotation = rotation
        self.fitresult = fitresult

        self.data = data.transform_to(self.frame)

    # /def

    @property
    def origin(self):
        return self._origin

    # /def

    @property
    def rotation(self):
        return self._rotation

    # /def

    @property
    def fit_values(self):
        return MappingProxyType(
            dict(origin=self.origin, rotation=self.rotation),
        )

    # /def

    @lazyproperty
    def frame(self):
        """SkyOffsetFrame."""
        # make frame  # TODO ensure same as `make_frame`
        frame = coord.SkyOffsetFrame(**self.fit_values)
        frame.differential_type = coord.SphericalCosLatDifferential
        return frame

    # /def

    @property
    def residual(self):
        return np.abs(self.data.lat - 0.0)

    # /def

    @property
    def residual_scalar(self):
        return np.sum(self.residual)

    # /def

    @lazyproperty
    def lon_order(self):
        """Order data by longitude.

        Returns
        -------
        order : ndarray

        """
        orderer = np.argsort(self.data.lon)

        return orderer

    # /def

    # ---------------------

    def __repr__(self):
        return f"FitResult({self.fit_values})"

    # /def

    # ---------------------

    def plot_data(self):
        # THIRD PARTY
        import matplotlib.pyplot as plt

        plt.scatter(self.data.lon, self.data.lat)
        plt.ylim(-90, 90)

        # return fig

    # /def

    def plot_on_residual(self, scalar: bool = True):
        # THIRD PARTY
        import matplotlib.pyplot as plt

        if scalar:
            theta = self.fit_values["rotation"]
            # plt.axvline(theta)
            plt.scatter(theta, self.residual_scalar, c="r")

        else:
            raise NotImplementedError

    # /def


# /class


##############################################################################
# END
