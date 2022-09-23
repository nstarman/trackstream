"""Fit a rotated reference frame to stream data."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from dataclasses import dataclass
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any, Generic, Mapping, Sequence, TypeVar, cast, final

# THIRD PARTY
import astropy.coordinates as coords
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import OptimizeResult

# LOCAL
from trackstream.common import CollectionBase
from trackstream.utils.visualization import PlotDescriptorBase

if TYPE_CHECKING:
    # THIRD PARTY
    from astropy.units import Quantity
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    # LOCAL
    from trackstream.stream.base import StreamBase

__all__: list[str] = []


##############################################################################
# Parameters

R = TypeVar("R")


##############################################################################
# CODE
##############################################################################


@final
@dataclass(frozen=True)
class FrameOptimizeResultPlotDescriptor(PlotDescriptorBase["FrameOptimizeResult"]):
    """FrameOptimizeResult plot descriptor."""

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
        # LOCAL
        from trackstream.frame.fit import residual

        fr, _ax, *_ = self._setup(ax=ax)

        # Residual plot
        rotation_angles = np.linspace(-180, 180, num=1_000, dtype=float)
        r = fr.origin.data
        xyz = (
            stream.data_coords.represent_as(coords.UnitSphericalRepresentation)  # type: ignore
            .represent_as(coords.CartesianRepresentation)
            .xyz.value
        )
        res = np.array(
            [
                residual((float(angle), float(r.lon.deg), float(r.lat.deg)), xyz, scalar=True)
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

    # ===============================================================

    def _in_own_frame(
        self,
        stream: StreamBase,
        origin: bool,
        *,
        axs: Sequence[Axes],
        format_ax: bool,
        set_title: bool,
        plot_vs: bool,
        kwargs: Mapping[str, Any],
    ) -> None:
        stream.plot.in_frame(frame="icrs", kind="positions", ax=axs[0], origin=origin, **kwargs)
        if format_ax and set_title:
            axs[0].set_title("Stream Star Positions")

        if plot_vs:
            ORIGIN_HAS_VS = "s" in stream.origin.data.differentials  # type: ignore

            stream.plot.in_frame(frame="icrs", kind="kinematics", ax=axs[1], origin=origin and ORIGIN_HAS_VS, **kwargs)
            if format_ax and set_title:
                axs[1].set_title("Stream Star Kinematics")

    def _in_rotated_frame(
        self,
        stream: StreamBase,
        fr: FrameOptimizeResult,
        *,
        origin: bool,
        kwargs: dict[str, Any],
        axs: Sequence[Axes],
        plot_vs: bool,
    ) -> None:
        kwargs.pop("c", None)  # setting from residual
        c: Quantity | dict[str, Quantity]
        rotstr = r" ($\theta=$" + f"{fr.rotation.value:.4}" + r"$^\degree$)"
        full_name = stream.full_name or ""

        if isinstance(stream, CollectionBase):
            c = {k: fr.calculate_residual(arm.coords) for k, arm in stream.items()}
            label = {k: arm.name + rotstr for k, arm in stream.items()}
        else:
            c = fr.calculate_residual(stream.coords)
            label = full_name + rotstr
        kwargs.pop("label", None)  # setting from residual
        stream.plot.in_frame(frame="stream", kind="positions", c=c, ax=axs[0], label=label, origin=origin, **kwargs)

        if plot_vs:
            ORIGIN_HAS_VS = "s" in stream.origin.data.differentials  # type: ignore

            stream.plot.in_frame(
                frame="stream",
                kind="kinematics",
                c=c,
                ax=axs[1],
                label=label,
                origin=origin and ORIGIN_HAS_VS,
                **kwargs,
            )

    def multipanel(
        self,
        stream: StreamBase,
        *,
        origin: bool = True,
        axes: np.ndarray | None = None,
        format_ax: bool = True,
        **kwargs: Any,
    ) -> tuple[Figure, np.ndarray]:
        """Plot frame fit in a 3 panel plot.

        - plot 1 : stream in its own frame
        - plot 2 : residual
        - plot 3 : rotated stream

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
        ValueError
            If the stream does not have a fit frame.
        """
        # Plot setup
        axs: np.ndarray
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
        self._in_own_frame(
            stream=stream,
            origin=origin,
            axs=tuple(axs[0, :]),
            format_ax=format_ax,
            set_title=set_title,
            plot_vs=plot_vs,
            kwargs=kwargs,
        )

        # Plot 2 : Residual
        self.residual(stream, ax=axs[1, 0], format_ax=format_ax)
        if format_ax and set_title:
            axs[1, 0].set_title("Frame Residual")
        if plot_vs:
            axs[1, 1].set_axis_off()

        # Plot 3 : Rotated Stream
        # Here it matters the length of the stream to the plotting method.
        self._in_rotated_frame(
            stream=stream, fr=self.enclosing, kwargs=kwargs, plot_vs=plot_vs, axs=tuple(axs[2, :]), origin=origin
        )

        return fig, axs


# ===================================================================


@final
@dataclass(frozen=True, repr=True)
class FrameOptimizeResult(Generic[R]):
    """Result of fitting a rotated frame.

    Parameters
    ----------
    frame : |Frame|
        The fit frame.
    result : object
        Fit results, e.g. `~scipy.optimize.OptimizeResult` if the frame was fit
        using :mod:`scipy`.
    """

    plot = FrameOptimizeResultPlotDescriptor()

    # ===============================================================

    frame: coords.SkyOffsetFrame
    result: R

    # ===============================================================

    @property
    def rotation(self) -> Quantity:
        """The rotation of point on sky."""
        return cast(u.Quantity, self.frame.rotation)

    @property
    def origin(self) -> coords.BaseCoordinateFrame:
        """The location of point on sky."""
        return cast(coords.BaseCoordinateFrame, self.frame.origin)

    # ===============================================================

    @singledispatchmethod
    @classmethod
    def from_result(
        cls: type[FrameOptimizeResult[Any]], optimize_result: object, frame: coords.BaseCoordinateFrame | None
    ) -> FrameOptimizeResult[R]:
        """Construct from object.

        Parameters
        ----------
        optimize_result : object
            Instantiation is single-dispatched on the object type.
        frame : Frame | None
            The fit frame.

        Returns
        -------
        FrameOptimizeResult
            With attribute ``result`` determed by ``optimize_result``.

        Raises
        ------
        NotImplementedError
            If there is no dispatched method.
        ValueError
            If the frame is not `None` and not equal to the frame in ``optimize_result``.
        """
        if not isinstance(optimize_result, cls):
            raise NotImplementedError(f"optimize_result type {type(optimize_result)} is not known.")

        # overload + Self is implemented here until it works
        if frame is not None and frame != optimize_result.frame:
            raise ValueError("frame must be None or the same as optimize_result's frame")
        return cls(frame=optimize_result.frame, result=optimize_result.result)

    @from_result.register(OptimizeResult)
    @classmethod
    def _from_result_scipyoptresult(
        cls: type[FrameOptimizeResult[Any]], optimize_result: OptimizeResult, frame: coords.BaseCoordinateFrame
    ) -> FrameOptimizeResult[OptimizeResult]:
        # Get coordinates
        optimize_result.x <<= u.deg
        fit_rot, fit_lon, fit_lat = optimize_result.x
        # create SkyCoord
        r = coords.UnitSphericalRepresentation(lon=fit_lon, lat=fit_lat)
        origin = coords.SkyCoord(frame.realize_frame(r, representation_type=frame.representation_type), copy=False)
        # transform to offset frame
        fit_frame = origin.skyoffset_frame(rotation=fit_rot)
        fit_frame.representation_type = frame.representation_type
        return cls(fit_frame, optimize_result)

    # ===============================================================

    def calculate_residual(self, data: coords.SkyCoord, scalar: bool = False) -> Quantity:
        """Calculate result residual given the fit frame.

        Parameters
        ----------
        data : (N,) `~astropy.coordinates.SkyCoord`
            The data for which to calculate the residual.
        scalar : bool
            Whether to sum the results to a scalar value.

        Returns
        -------
        Quantity
            Scalar if ``scalar``, else length N.
        """
        ur = data.transform_to(self.frame).represent_as(coords.UnitSphericalRepresentation)  # type: ignore
        res: Quantity = np.abs(ur.lat - 0.0 * u.rad)
        return np.sum(res) if scalar else res
