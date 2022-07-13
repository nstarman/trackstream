"""Fit a Rotated reference frame."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from copy import deepcopy
from types import FunctionType, MethodType
from typing import TYPE_CHECKING, Any, Sequence, TypedDict, TypeVar

# THIRD PARTY
import astropy.units as u
import matplotlib.pyplot as plt
import scipy.optimize as opt
from astropy.coordinates import BaseCoordinateFrame
from astropy.coordinates import CartesianRepresentation as CartRep
from astropy.coordinates import SkyCoord, SkyOffsetFrame, SphericalCosLatDifferential
from astropy.coordinates import UnitSphericalRepresentation as USphrRep
from astropy.units import Quantity
from attrs import define, field
from erfa import ufunc as erfa_ufunc
from matplotlib.figure import Figure
from matplotlib.pyplot import Axes
from numpy import abs, array, dot, linspace, median, ndarray, sqrt, square, sum
from scipy.optimize import Bounds

# LOCAL
from trackstream._typing import EllipsisType
from trackstream.base import CollectionBase, FramedBase
from trackstream.utils.coord_utils import reference_to_skyoffset_matrix
from trackstream.visualization import PlotDescriptorBase

if TYPE_CHECKING:
    # LOCAL
    from trackstream.stream.base import StreamBase

__all__ = ["RotatedFrameFitter", "residual"]


##############################################################################
# PARAMETERS

FT = TypeVar("FT", MethodType, FunctionType)


class _RotatedFrameFitterOptions(TypedDict):
    rot0: Quantity | None
    bounds: None | Sequence | Bounds
    # fix_origin: Optional[bool]
    leastsquares: bool | None
    align_v: bool | None


##############################################################################
# CODE
##############################################################################


def cartesian_model(
    data: CartRep,
    *,
    lon: Quantity | float,
    lat: Quantity | float,
    rotation: Quantity | float,
) -> tuple[Quantity, Quantity, Quantity]:
    """Model from Cartesian Coordinates.

    Parameters
    ----------
    data : |CartesianRep|
        Cartesian representation of the data.
    lon, lat : float or |Angle| or |Quantity| instance
        The |Longitude| and |Latitude| origin for the reference frame. If float,
        assumed degrees.
    rotation : float or |Angle| or |Quantity| instance
        The final rotation of the frame about the ``origin``. The sign of the
        rotation is the left-hand rule.  That is, an object at a particular
        position angle in the un-rotated system will be sent to the positive
        |Latitude| (z) direction in the final frame. If float, assumed degrees.

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


def residual(variables: tuple[float, float, float], data: CartRep, scalar: bool = False) -> float | ndarray:
    r"""How close phi2, the rotated |Latitude| (e.g. dec), is to flat.

    Parameters
    ----------
    variables : tuple[float, float, float]
        (rotation, lon, lat)

        - rotation angle : float
            The final rotation of the frame about the ``origin``. The sign of
            the rotation is the left-hand rule.  That is, an object at a
            particular position angle in the un-rotated system will be sent to
            the positive |Latitude| (z) direction in the final frame.
            In degrees.
        - lon, lat : float
            In degrees. If |ICRS|, equivalent to ra & dec.
    data : |CartesianRep|
        E.g. :attr:`astropy.coordinates.ICRS.cartesian`.

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


@define(frozen=True)
class RotatedFrameFitter(FramedBase):
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

    leastsquares : bool, optional, keyword-only
        Whether to to use :func:`~scipy.optimize.least_square` or
        :func:`~scipy.optimize.minimize`. Default is False.

    align_v : bool
        Whether to align by the velocity.
    """

    origin: SkyCoord
    """The location of point on sky about which to rotate."""

    default_fit_options: dict[str, Any] = field()
    """The fitting options."""

    minimizer_kwargs: dict[str, Any] = field(factory=dict)

    @default_fit_options.default  # type: ignore
    def _default_fit_options_factory(self):
        kw = dict.fromkeys(_RotatedFrameFitterOptions.__annotations__.keys())

        kw["bounds"] = self.make_bounds(
            rot_lower=Quantity(-180.0, u.deg),
            rot_upper=Quantity(180.0, u.deg),
            origin_lim=Quantity(0.005, u.deg),
        )
        return kw

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        # need to transform
        origin = self.origin.transform_to(self.frame)
        origin.representation_type = self.frame_representation_type
        object.__setattr__(self, "origin", origin)

    #######################################################

    def make_bounds(self, rot_lower: Quantity, rot_upper: Quantity, origin_lim: Quantity) -> Bounds:
        """Make bounds on Rotation parameter.

        Parameters
        ----------
        rot_lower, rot_upper : Quantity ['angle'], optional
            The lower and upper bounds in degrees.
        origin_lim : Quantity ['angle'], optional
            The symmetric lower and upper bounds on origin in degrees.

        Returns
        -------
        `~scipy.optimize.Bounds`
        """
        origin = self.origin.represent_as(USphrRep)
        bounds = Bounds(
            Quantity([rot_lower, origin.lon - origin_lim, origin.lat - origin_lim], u.deg),
            Quantity([rot_upper, origin.lon + origin_lim, origin.lat + origin_lim], u.deg),
        )

        return bounds

    def align_v_positive_lon(
        self,
        data: SkyCoord,
        fit_values: dict[str, Any],
        subsel: EllipsisType | Sequence | slice = Ellipsis,  # type: ignore
    ) -> dict[str, Any]:
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
        rot0: Quantity | None = None,
        bounds: ndarray | None = None,
        *,
        # fix_origin: Optional[bool] = None,
        leastsquares: bool | None = None,
        align_v: bool | None = None,  # TODO!
        **minimizer_kwargs: Any,
    ) -> FrameOptimizeResult:
        """Find Best-Fit Rotated Frame.

        Parameters
        ----------
        data : `astropy.coordinates.SkyCoord`, positional only rot0 : Quantity,
        optional
            Initial guess for rotation
        bounds : sequence or `~scipy.optimize.Bounds`, optional
            Bounds on variables. Must be values in units of degrees. There are
            two ways to specify the bounds:

            1. Instance of `~scipy.optimize.Bounds` class.
            2. Sequence of ``(min, max)`` pairs for each of ``x=[rotation,
               origin_lon, origin_lat``. `None` is used to specify no bound.

        Returns
        -------
        `~trackstream.fit.rotated_frame.FrameOptimizeResult`

        Other Parameters
        ----------------
        leastsquares : bool, optional, keyword-only
            Whether to to use :func:`~scipy.optimize.least_square` or
            :func:`~scipy.optimize.minimize` (default).
        align_v : bool, optional, keyword-only
            Whether to align velocity to be in positive direction
        minimizer_kwargs:
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

        # minimizer kwargs, preferring newer
        kwargs: dict[str, Any] = {**self.minimizer_kwargs, **minimizer_kwargs}

        rot0 = self.default_fit_options["rot0"] if rot0 is None else rot0
        if rot0 is None:
            raise ValueError("no prespecified `rot0`; need to provide one.")

        # Process fit options
        leastsquares = self.default_fit_options["leastsquares"] if leastsquares is None else leastsquares
        if leastsquares:
            minimizer = opt.least_squares
            method = kwargs.pop("method", "trf")
        else:
            minimizer = opt.minimize
            method = kwargs.pop("method", "slsqp")

        # -----------------------------
        # Fit

        x0 = Quantity([rot0, origin.lon, origin.lat], u.deg).value

        fit_result: opt.OptimizeResult = minimizer(
            residual, x0=x0, args=(rep, not leastsquares), method=method, bounds=bounds, **kwargs
        )
        # fit_rot, fit_lon, fit_lat = fit_result.x << u.deg

        # TODO!
        # values = dict(rotation=fit_rot, origin=fit_origin)
        # if align_v is None:
        #     align_v = self.default_fit_options["align_v"]
        # if align_v is None and "s" in data.data.differentials:
        #     align_v = True
        # if align_v:
        #     values = self.align_v_positive_lon(values, subsel=...)

        return FrameOptimizeResult.from_result(
            fit_result,
            frame=self.frame,
            # representation_type=self.frame_representation_type,
        )


# -------------------------------------------------------------------


class FrameOptimizeResultPlotDescriptor(PlotDescriptorBase["FrameOptimizeResult"]):
    """Plot FrameOptimizeResult."""

    def residual(self, stream: StreamBase, *, ax: Axes | None = None, format_ax: bool = True) -> Axes:
        """Residual plot of fitting the frame.

        Parameters
        ----------
        stream : `trackstream.stream.base.StreamBase`
            The stream for which to plot the residual.

        ax : |Axes| or None, optional keyword-only
            Matplotlib |Axes|. `None` (default) uses the current axes
            (:func:`matplotlib.pyplot.gca`).
        format_ax : bool, optional keyword-only
            Whether to add the axes labels and info, by default `True`.

        Returns
        -------
        |Axes|
        """
        fr, _ax, *_ = self._setup(ax=ax)

        # Residual plot
        rotation_angles: ndarray = linspace(-180, 180, num=1_000, dtype=float)
        r = fr.origin.data
        res = array(
            [
                residual(
                    (float(angle), float(r.lon.deg), float(r.lat.deg)),
                    stream.data_coords.cartesian,
                    scalar=True,
                )
                for angle in rotation_angles
            ]
        )
        _ax.scatter(rotation_angles, res)

        # Plot the best-fit rotation
        _ax.axvline(
            fr.rotation.value,
            c="k",
            ls="--",
            label=f"best-fit rotation = {fr.rotation.value:.2f}" + r"$^\degree$",
        )
        # and the next period
        next_period = 180 if (fr.rotation.value - 180) < rotation_angles.min() else -180
        _ax.axvline(fr.rotation.value + next_period, c="k", ls="--", alpha=0.5)

        if format_ax:
            _ax.set_xlabel(r"Rotation angle $\theta$", fontsize=13)
            _ax.set_ylabel(r"Residual / # data pts", fontsize=13)
            _ax.legend(loc="lower left")

        return _ax

    def multipanel(
        self,
        stream: StreamBase,
        *,
        origin: bool = True,
        axes: ndarray | None = None,
        format_ax: bool = True,
        **kwargs: Any,
    ) -> tuple[Figure, ndarray]:
        """Plot frame fit in a 3 panel plot.

        Parameters
        ----------
        origin : bool, optional keyword-only
            Whether to plot the origin, by default `True`.

        axes : ndarray[|Axes|] or None, optional
            Matplotlib |Axes|. `None` (default) makes a new |Figure| and |Axes|.
        format_ax : bool, optional keyword-only
            Whether to add the axes labels and info, by default `True`.

        **kwargs : Any
            Passed to ``in_frame`` for both positions and kinematics (if
            present).

        Returns
        -------
        |Figure|
            The matplotlib figure.
        ndarray[|Axes|]
            The matplotlib figure axes.

        Raises
        ------
        Exception
            If the stream does not have a fit frame.
        """
        fr = self._enclosing
        ORIGIN_HAS_VS = "s" in stream.origin.data.differentials
        full_name = stream.full_name or ""

        # Plot setup
        axs: ndarray
        if axes is not None:
            plot_vs = axes.ndim > 1 and axes.shape[1] >= 2 and stream.has_kinematics
            axs = axes.reshape(-1, 1) if axes.ndim == 1 else axes
            fig = axs.flat[0].figure  # assuming all the same
        else:
            plot_vs = stream.has_kinematics
            nrows = 2 if plot_vs else 1
            figsize = (16 if plot_vs else 8, 7)
            fig, axs = plt.subplots(3, nrows, figsize=figsize)  # type: ignore

        set_title: bool = kwargs.pop("title", True)
        kwargs.setdefault("format_ax", format_ax)
        origin = kwargs.pop("origin", kwargs.setdefault("origin", origin))

        # Plot 1 : Stream in its own frame
        stream.plot.in_frame(frame="ICRS", kind="positions", ax=axs[0, 0], origin=origin, **kwargs)
        if format_ax and set_title:
            axs[0, 0].set_title("Stream Star Positions")
        if plot_vs:
            stream.plot.in_frame(
                frame="ICRS",
                kind="kinematics",
                ax=axs[0, 1],
                origin=origin and ORIGIN_HAS_VS,
                **kwargs,
            )
            if format_ax and set_title:
                axs[0, 1].set_title("Stream Star Kinematics")

        # Plot 2 : Residual
        # self._fit_frame_residual(ax=axs[1, 0], format_ax=format_ax)
        self.residual(stream, ax=axs[1, 0], format_ax=format_ax)
        if format_ax and set_title:
            axs[1, 0].set_title("Frame Residual")
        if plot_vs:
            axs[1, 1].set_axis_off()

        # Plot 3 : Rotated Stream
        # here it matters the length of the stream to the plotting method.
        kwargs.pop("c", None)  # setting from residual
        c: Quantity | dict[str, Quantity]
        rotstr = r" ($\theta=$" + f"{fr.rotation.value:.4}" + r"$^\degree$)"
        if isinstance(stream, CollectionBase):
            c = {k: fr.calculate_residual(arm.coords_ord) for k, arm in stream.items()}
            label = {k: arm.name + rotstr for k, arm in stream.items()}
        else:
            c = fr.calculate_residual(stream.coords_ord)
            label = full_name + rotstr
        kwargs.pop("label", None)  # setting from residual
        stream.plot.in_frame(
            frame="stream",
            kind="positions",
            c=c,
            ax=axs[2, 0],
            label=label,
            origin=origin,
            **kwargs,
        )
        if plot_vs:
            stream.plot.in_frame(
                frame="stream",
                kind="kinematics",
                c=c,
                ax=axs[2, 1],
                label=label,
                origin=origin and ORIGIN_HAS_VS,
                **kwargs,
            )

        return fig, axs


@define(frozen=True, slots=False, kw_only=True, repr=False)
class FrameOptimizeResult(opt.OptimizeResult, FramedBase):
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

    plot = FrameOptimizeResultPlotDescriptor()

    def __init__(self, *, frame: SkyOffsetFrame, **kwargs):
        super().__init__(**kwargs)
        self.__attrs_init__(frame=frame)

    @classmethod
    def from_result(cls, optimize_result: opt.OptimizeResult, frame: BaseCoordinateFrame):
        optimize_result.x <<= u.deg
        fit_rot, fit_lon, fit_lat = optimize_result.x
        r = USphrRep(lon=fit_lon, lat=fit_lat)
        origin = SkyCoord(frame.realize_frame(r, representation_type=frame.representation_type), copy=False)
        fit_frame = origin.skyoffset_frame(rotation=fit_rot)
        fit_frame.representation_type = frame.representation_type
        return cls(frame=fit_frame, **optimize_result)

    @property
    def rotation(self) -> Quantity:
        """The rotation of point on sky."""
        return self.frame.rotation

    @property
    def origin(self) -> BaseCoordinateFrame:
        """The location of point on sky."""
        return self.frame.origin

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

    def __repr__(self) -> str:
        s = object.__repr__(self) + "\n" + super().__repr__()
        return s
