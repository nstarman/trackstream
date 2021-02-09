# -*- coding: utf-8 -*-

"""Fit a Rotated ICRS reference frame.

.. todo::

    Use an Astropy Model instead.


"""


__all__ = [
    "RotatedFrameFitter",
    "cartesian_model",
    "residual",
    "make_bounds",
    "fit_frame",
    "align_v_positive_lon",
]


##############################################################################
# IMPORTS

# BUILT-IN
import copy
import typing as T
from types import MappingProxyType

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import scipy.optimize as opt
from astropy.utils.decorators import format_doc, lazyproperty

# FIRST PARTY

# PROJECT-SPECIFIC
from .utils import cartesian_to_spherical, reference_to_skyoffset_matrix
from trackstream.config import conf
from trackstream.setup_package import HAS_LMFIT
from trackstream.type_hints import QuantityType

if HAS_LMFIT:
    # FIRST PARTY
    import utilipy as lf
    from utilipy.data_utils.fitting import scipy_residual_to_lmfit

    scipy_residual_to_lmfit_dec = scipy_residual_to_lmfit.decorator

else:
    scipy_residual_to_lmfit_dec = (
        lambda param_order: lambda x: x
    )  # noqa: E7301

##############################################################################
# CODE
##############################################################################


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


@scipy_residual_to_lmfit_dec(param_order=["rotation", "lon", "lat"])
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

_make_bounds_defaults = dict(
    rot_lower=-180.0 * u.deg,
    rot_upper=180.0 * u.deg,
    origin_lim=0.005 * u.deg,
)


# @u.quantity_input(rot_lower=u.deg, rot_upper=u.deg, origin_lim=u.deg)
def make_bounds(
    origin: coord.UnitSphericalRepresentation,
    rot_lower: u.Quantity = _make_bounds_defaults["rot_lower"],
    rot_upper: u.Quantity = _make_bounds_defaults["rot_upper"],
    origin_lim: u.Quantity = _make_bounds_defaults["origin_lim"],
) -> T.Tuple[float, float]:
    """Make bounds on Rotation parameter.

    Parameters
    ----------
    rot_lower, rot_upper : |Quantity|, optional
        The lower and upper bounds in degrees.
    origin_lim : |Quantity|, optional
        The symmetric lower and upper bounds on origin in degrees.

    Returns
    -------
    bounds : ndarray
        Shape (3, 2)
        Rows are rotation_bounds, lon_bounds, lat_bounds

    """
    rotation_bounds = (rot_lower.to_value(u.deg), rot_upper.to_value(u.deg))
    # longitude bounds (ra in ICRS).
    lon_bounds = (origin.lon + (-1, 1) * origin_lim).to_value(u.deg)
    # latitude bounds (dec in ICRS).
    lat_bounds = (origin.lat + (-1, 1) * origin_lim).to_value(u.deg)

    # stack bounds so rows are bounds.
    bounds = np.c_[rotation_bounds, lon_bounds, lat_bounds].T

    return bounds


# /def


# -------------------------------------------------------------------


def _fit_representation_lmfit(
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


def _fit_representation_scipy(
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


_minimize_defaults = dict(
    fix_origin=False,
    use_lmfit=None,
    leastsquares=False,
    align_v=True,
)
"""Default values for `~trackstream.preprocess.fit_rotated_frame.minimize`."""


# @u.quantity_input(rot0=u.deg)
def fit_frame(
    data: coord.BaseCoordinateFrame,
    origin: coord.BaseCoordinateFrame,
    rot0: u.Quantity = 0 * u.deg,
    bounds: T.Sequence = (-np.inf, np.inf),
    *,
    fix_origin: bool = _minimize_defaults["fix_origin"],
    use_lmfit: T.Optional[bool] = _minimize_defaults["use_lmfit"],
    leastsquares: bool = _minimize_defaults["leastsquares"],
    align_v: bool = _minimize_defaults["align_v"],
    **fit_kwargs,
):
    """Find Best-Fit Rotated Frame.

    Parameters
    ----------
    data : :class:`~astropy.coordinates.CartesianRepresentation`
        If `align_v`, must have attached differentials

    rot0 : |Quantity|
        Initial guess for rotation
    origin : :class:`~astropy.coordinates.BaseCoordinateFrame`
        location of point on sky about which to rotate

    bounds : array-like, optional
        Parameter bounds.
        See :func:`~trackstream.preprocess.fit_rotated_frame.make_bounds`
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
    use_lmfit : bool, optional, kwarg only
        Whether to use ``lmfit`` package
    leastsquares : bool, optional, kwarg only
        If `use_lmfit` is False, whether to to use
        :func:`~scipy.optimize.least_square` or
        :func:`~scipy.optimize.minimize` (default)

    align_v : bool, optional, kwarg only
        Whether to align velocity to be in positive direction

    fit_kwargs:
        Into whatever minimization package / function is used.

    Raises
    ------
    ValueError
        If `use_lmfit` and lmfit is not installed.

    """
    # ------------------------
    # Inputs

    # Data
    # need to make sure Cartesian representation
    # data = data.represent_as(
    #     coord.CartesianRepresentation,
    #     differential_class=coord.CartesianDifferential,
    # )

    # Origin
    # We work with a SphericalRepresentation, but
    # if isinstance(origin, coord.SkyCoord):
    #     raise TypeError
    origin_frame = origin.__class__
    origin = origin.represent_as(coord.SphericalRepresentation)

    if use_lmfit is None:
        use_lmfit = conf.use_lmfit

    x0 = u.Quantity([rot0, origin.lon, origin.lat]).to_value(u.deg)
    subsel = fit_kwargs.pop("subsel", Ellipsis)

    # ------------------------
    # Fitting

    if use_lmfit:  # lmfit
        if not _HAS_LMFIT:
            raise ValueError("`lmfit` package not available.")

        res, values = _fit_representation_lmfit(
            data.cartesian,
            x0=x0,
            bounds=bounds,
            fix_origin=fix_origin,
            **fit_kwargs,
        )

    else:  # scipy
        res, values = _fit_representation_scipy(
            data.cartesian,
            x0=x0,
            bounds=bounds,
            fix_origin=fix_origin,
            use_leastsquares=leastsquares,
            **fit_kwargs,
        )

    # /def

    # ------------------------
    # Return

    best_rot = values[0]
    best_origin = coord.UnitSphericalRepresentation(
        lon=values[1],
        lat=values[2],  # TODO re-add distance
    )
    best_origin = origin_frame(best_origin)

    values = dict(rotation=best_rot, origin=best_origin)
    if align_v:
        values = align_v_positive_lon(data, values, subsel=subsel)

    return res, values


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


def order_data_from_lon(data: coord.BaseCoordinateFrame) -> np.ndarray:
    """Order data by Longitude.

    Parameters
    ----------
    data : `~astropy.coordinates.BaseCoordinateFrame`
        Must be output of SkyOffsetFrame.

    Returns
    -------
    order : ndarray

    """
    arr = np.arange(len(data))
    orderer = np.argsort(data.lon)

    return arr[orderer]


# /def


#####################################################################


@format_doc(None, **{**_make_bounds_defaults, **_minimize_defaults})
class RotatedFrameFitter(object):
    """Class to Fit Rotated Frames.

    Parameters
    ----------
    data : :class:`~astropy.coordinates.BaseCoordinateFrame`
        In ICRS coordinates.

    origin : :class:`~astropy.coordinates.ICRS`
        location of point on sky about which to rotate.

    Other Parameters
    ----------------
    rot_lower, rot_upper : |Quantity|, optional, kwarg only
        The lower and upper bounds in degrees.
        Default is ({rot_lower}, {rot_upper}).
    origin_lim : |Quantity|, optional, kwarg only
        The symmetric lower and upper bounds on origin in degrees.
        Default is {origin_lim}.

    fix_origin : bool, optional, kwarg only
        Default is {fix_origin}
    use_lmfit : bool, optional, kwarg only
        Whether to use ``lmfit`` package. Default is {use_lmfit}
    leastsquares : bool, optional, kwarg only
        If `use_lmfit` is False, whether to to use
        :func:`~scipy.optimize.least_square` or
        :func:`~scipy.optimize.minimize`
        Default is {leastsquares}

    """

    def __init__(
        self, data: coord.CartesianRepresentation, origin: coord.ICRS, **kwargs
    ):
        super().__init__()
        # Store Data & Origin
        self.data = data
        self.origin = origin

        # Create bounds
        # pop rot_lower, rot_upper, origin_lim from kwargs
        # if not in kwargs, get from _make_bounds_defaults
        bounds_args = {
            k: kwargs.pop(k, v) for k, v in _make_bounds_defaults.items()
        }
        self.make_bounds(**bounds_args)

        # Minimizer kwargs (reverse order of precedence)
        self.fitter_kwargs = {**_minimize_defaults, **kwargs}
        if self.fitter_kwargs["use_lmfit"] is None:  # get from config
            self.fitter_kwargs["use_lmfit"] = conf.use_lmfit

        # determine whether velocity exists to break +/- 180 degree degeneracy
        # If it does, call the `align_v` option in `fit_frame`
        if "s" in self.data.data.differentials:
            self.fitter_kwargs["align_v"] = True  # force true as default
        else:
            self.fitter_kwargs["align_v"] = False

    # /def

    # @u.quantity_input(rot_lower=u.deg, rot_upper=u.deg, origin_lim=u.deg)
    def make_bounds(
        self,
        rot_lower=_make_bounds_defaults["rot_lower"],
        rot_upper=_make_bounds_defaults["rot_upper"],
        origin_lim=_make_bounds_defaults["origin_lim"],
    ) -> T.Tuple[float, float]:
        """Make bounds on Rotation parameter.

        Parameters
        ----------
        rot_lower, rot_upper : |Quantity|, optional
            The lower and upper bounds in degrees.
        origin_lim : |Quantity|, optional
            The symmetric lower and upper bounds on origin in degrees.

        """
        self.bounds = make_bounds(
            self.origin.data,  # unitspherical form
            rot_lower=rot_lower,
            rot_upper=rot_upper,
            origin_lim=origin_lim,
        )

    # /def

    # @u.quantity_input(rot0=u.deg)
    def fit(
        self,
        rot0: T.Optional[u.Quantity] = None,
        bounds: T.Optional[T.Sequence] = None,
        **kwargs,
    ):
        if rot0 is None:
            if "rot0" not in self.fitter_kwargs:
                raise ValueError(
                    "No prespecified `rot0`; " "Need to provide one.",
                )
            rot0 = self.fitter_kwargs["rot0"]

        if bounds is None:
            bounds = self.bounds

        kwargs = {**self.fitter_kwargs, **kwargs}

        fit_result, fit_values = fit_frame(
            self.data, origin=self.origin, bounds=bounds, **kwargs
        )

        return FitResult(self.data, fitresult=fit_result, **fit_values)

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
        # PROJECT-SPECIFIC
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
    data : |CoordinateFrame|
        In ICRS coordinates.
    fit_values : Dict[str, Any]
        Has keys "rotation" and "origin".
    fitresult : Any, optional

    Attributes
    ----------
    data : |CoordinateFrame|
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
        arr = np.arange(len(self.data))
        orderer = np.argsort(self.data.lon)

        return arr[orderer]

    # /def

    # ---------------------

    def __repr__(self):
        return f"FitResult({self.fit_values})"

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
