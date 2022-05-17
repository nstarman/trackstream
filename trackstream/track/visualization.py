# -*- coding: utf-8 -*-

"""Stream track plotting."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

# THIRD PARTY
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.units import Quantity
from matplotlib.axes import Axes
from matplotlib.collections import EllipseCollection
from matplotlib.figure import Figure
from numpy import arange, array, atleast_1d, atleast_2d, ndarray
from astropy.utils.decorators import format_doc

# LOCAL
from .rotated_frame import FrameOptimizeResult
from trackstream._type_hints import FrameLikeType
from trackstream.utils.misc import covariance_ellipse
from trackstream.visualization import _DS, _DSf, CLike, DKindT, StreamPlotDescriptorBase

if TYPE_CHECKING:
    # LOCAL
    from trackstream.track.fitresult import StreamTrack  # noqa: F401


__all__: List[str] = []


##############################################################################
# CODE
##############################################################################


class StreamTrackPlotDescriptor(StreamPlotDescriptorBase["StreamTrack"]):
    def _setup(self, ax: Optional[Axes]) -> Tuple[Any, ...]:
        """Setup the plot.

        Parameters
        ----------
        ax : |Axes|

        Returns
        -------
        tuple[Any, ...]
            At least (`trackstream.visualization.NamedWithCoords`, |Axes|)
        """
        track, ax = super()._setup(ax=ax)
        full_name = track.full_name or ""

        stream = track.stream
        if stream is None:
            raise ValueError

        return track, ax, stream, full_name

    # ===============================================================
    # Individual plot methods

    @format_doc(None, frame=_DSf["frame"](3), DKindT=_DS["DKindT"], dkind_ds=_DS["dkind_ds"])
    def in_frame(
        self,
        frame: str = "stream",
        kind: DKindT = "positions",
        *,
        plot_origin: bool = True,
        ax: Optional[Axes] = None,
        c: CLike = "tab:blue",
        format_ax: bool = False,
        **kwargs: Any,
    ) -> Axes:
        """Plot stream in a stream frame.

        Parameters
        ----------
        frame : |Frame| or str, optional
            {frame}
        kind : {DKindT}, optional
            {dkind_ds}

        c : str or array-like[float], optional
            The color or sequence thereof, by default "tab:blue"
        plot_origin : bool, optional
            Whether to plot the origin, by default `True`
        ax : Optional[|Axes|], optional
            Matplotlib |Axes|, by default `None`
        format_ax : bool, optional
            Whether to add the axes labels and info, by default `True`.

        Returns
        -------
        |Axes|
        """
        _, _ax, stream, *_ = self._setup(ax)

        super().in_frame(frame=frame, kind=kind, c=c, ax=_ax, format_ax=format_ax, **kwargs)
        # TODO! by arm

        if plot_origin:
            self.origin(stream.origin, frame=frame, kind=kind, ax=_ax)

        return _ax

    def fit_frame_residual(self, *, ax: Optional[Axes] = None, format_ax: bool = True) -> Axes:
        """Residual plot of fitting the frame.

        Parameters
        ----------
        ax : Optional[Axes], optional
            Matplotlib |Axes|, by default None
        format_ax : bool, optional
            Whether to add the axes labels and info, by default True

        Returns
        -------
        |Axes|

        Raises
        ------
        Exception
            If the stream is not fit.
        """
        track, _ax, stream, *_ = self._setup(ax)

        fr: Optional[FrameOptimizeResult] = track.frame_fit
        if fr is None:
            raise Exception("need to fit the stream first")

        if format_ax:
            _ax.set_title("Frame Residual")

        return fr.plot.residual(stream, ax=_ax, format_ax=format_ax)

    @format_doc(None, frame=_DSf["frame"](3), DKindT=_DS["DKindT"], dkind_ds=_DS["dkind_ds"])
    def som(
        self,
        frame: FrameLikeType = "stream",
        kind: DKindT = "positions",
        *,
        original_prototypes: bool = False,
        y_offset: Quantity = Quantity(0.0, u.deg),
        ax: Optional[Axes] = None,
        format_ax: bool = False,
        plot_origin: bool = True,
    ) -> Axes:
        """Plot Self-Organizing Map.

        Parameters
        ----------
        frame : |Frame| or str, optional
            {frame}
        kind : {DKindT}, optional
            {dkind_ds}

        original_prototypes : bool, optional keyword-only
            Whether to plot the original prototypes, by default `False`
        y_offset : Quantity, optional keyword-only
            Offset in latitude (y coordinate) for the prototypes.
        ax : |Axes| or None, optional keyword-only
            Matplotlib |Axes|, by default `None`
        format_ax : bool, optional keyword-only
            Whether to add the axes labels and info, by default `True`

        Returns
        -------
        |Axes|
        """
        if frame.lower() != "stream":
            raise NotImplementedError
        elif kind != "positions":
            raise ValueError("SOM is only run on positions")

        track, _ax, stream, _, *_ = self._setup(ax)
        theframe, frame_name = self._parse_frame(frame)
        soms = cast(dict, getattr(track, "som", {}))
        xn, yn = "", ""  # default labels. updated from data.

        # raise error if there's nothing to plot
        arm1 = stream.arm1
        som1 = soms.get("arm1")
        arm2 = stream.arm2
        som2 = soms.get("arm2")
        if (not arm1.has_data or som1 is None) and (not arm2.has_data or som2 is None):
            raise ValueError("cannot plot som without data")

        # Plot arm 1
        if arm1.has_data and som1 is not None:
            ps1, _ = self._to_frame(som1.prototypes, theframe)
            (x, xn), (y, yn) = self._get_xy(ps1, kind=kind)

            _ax.scatter(
                x,
                y + y_offset,
                marker="P",  # type: ignore
                edgecolors="black",
                facecolor="none",
            )

            if original_prototypes:
                ips1, _ = self._to_frame(som1.init_prototypes, theframe)
                (x, xn), (y, yn) = self._get_xy(ips1, kind=kind)

                _ax.scatter(
                    x,
                    y - y_offset,  # type: ignore
                    marker="X",  # type: ignore
                    edgecolors="gray",
                    facecolor="none",
                )

        # Plot arm 2
        if stream.arm2.has_data and (som2 := soms.get("arm2")) is not None:
            ps2, _ = self._to_frame(som2.prototypes, theframe)
            (x, xn), (y, yn) = self._get_xy(ps2, kind=kind)
            _ax.scatter(
                x,
                y + y_offset,
                marker="P",  # type: ignore
                edgecolors="black",
                facecolor="none",
            )

            if original_prototypes:
                ips2, _ = self._to_frame(som2.init_prototypes, theframe)
                (x, xn), (y, yn) = self._get_xy(ips2, kind=kind)
                _ax.scatter(
                    x,
                    y - y_offset,  # type: ignore
                    marker="X",  # type: ignore
                    edgecolors="gray",
                    facecolor="none",
                )

        if plot_origin:
            self.origin(stream.origin, frame=frame, kind=kind, ax=_ax)

        if format_ax:  # Axes settings
            self._format_ax(_ax, frame=frame_name, x=xn, y=yn)

        return _ax

    # --------------------------------

    def _cov(
        self,
        ax: Axes,
        mean: ndarray,
        cov: ndarray,
        std: float = 1,
        *,
        facecolor: str = "gray",
        edgecolor: str = "none",
        alpha: float = 0.5,
        ls: str = "solid",
    ) -> None:
        angle, wh = covariance_ellipse(cov)
        width = 2 * atleast_1d(wh[..., 0])  # TODO! why 2?
        height = 2 * atleast_1d(wh[..., 0])

        ec = EllipseCollection(
            std * width,
            std * height,
            angle.to_value(u.deg),
            units="x",
            offsets=atleast_2d(mean),
            transOffset=ax.transData,
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
            lw=2,
            ls=ls,
        )
        ax.add_collection(ec)

        # for mn, a, w, h in zip(atleast_2d(mean), angle.to_value(u.deg), width, height):

        #     e = Ellipse(
        #         xy=mn,
        #         width=std * w,
        #         height=std * h,
        #         angle=a,
        #         facecolor=facecolor,
        #         edgecolor=edgecolor,
        #         alpha=alpha,
        #         lw=2,
        #         ls=ls,
        #     )
        #     ax.add_patch(e)

        # if show_center:
        #     x, y = mean
        #     plt.scatter(x, y, marker="+", color=edgecolor)  # type: ignore

    @format_doc(None, frame=_DSf["frame"](3), DKindT=_DS["DKindT"], dkind_ds=_DS["dkind_ds"])
    def kalman(
        self,
        frame: FrameLikeType = "stream",
        kind: DKindT = "positions",
        *,
        ax: Optional[Axes] = None,
        format_ax: bool = False,
        plot_origin: bool = False,
        **kwargs: Any,
    ) -> Axes:
        """Plot Kalman Filter.

        Parameters
        ----------
        frame : |Frame| or str, optional
            {frame}
        kind : {DKindT}, optional
            {dkind_ds}

        ax : Optional[Axes], optional
            matplotlib axes, by default `None`.

        Returns
        -------
        Axes
            matplotlib axes.
        """
        if frame.lower() != "stream":
            raise NotImplementedError
        else:
            _, frame_name = self._parse_frame(frame)

        track, _ax, stream, _, *_ = self._setup(ax)
        xn, yn = "", ""  # default, updated from data

        # TODO! allow for eval by `affine` using the Path
        # TODO! fill_between
        # pm = track()
        # ax.fill_between(
        #     pm.mean.lon[:, 0],
        #     y1=pm.mean.lat[:, 0] - pm.width["lat"] / 2,
        #     y2=pm.mean.lat[:, 0] + pm.width["lat"] / 2,
        # )

        # Subselect from covariance matrix
        if kind == "positions":
            start, stop = 0, 4
        elif kind == "kinematics":
            start, stop = 4, 8
        slc = (slice(None), slice(start, stop, 2), slice(start, stop, 2))

        if stream.arm1.has_data:
            # centers
            crd = track.kalman["arm1"]._v_to_crd(array(track.kalman["arm1"]._result.Xs[:, ::2]))
            (x, xn), (y, yn) = self._get_xy(crd, kind=kind)
            # error
            Ps = track.kalman["arm1"]._result.Ps[slc]

            self._cov(
                _ax,
                mean=array((x, y)).reshape((2, -1)).T[::5],
                cov=Ps[::5],
                **kwargs,
            )

        if stream.arm2.has_data:
            # centers
            crd = track.kalman["arm2"]._v_to_crd(array(track.kalman["arm2"]._result.Xs[:, ::2]))
            (x, xn), (y, yn) = self._get_xy(crd, kind=kind)
            # errors
            Ps = track.kalman["arm2"]._result.Ps[slc]

            self._cov(
                _ax,
                mean=array((x, y)).reshape((2, -1)).T[::5],
                cov=Ps[::5],
                **kwargs,
            )

        if plot_origin:
            self.origin(stream.origin, frame=frame, kind=kind, ax=_ax)

        if format_ax:
            self._format_ax(_ax, frame=frame_name, x=xn, y=yn)

        return _ax

    # ---------------------------------------------------------------

    @format_doc(None, frame=_DSf["frame"](3), DKindT=_DS["DKindT"], dkind_ds=_DS["dkind_ds"])
    def plot(
        self,
        frame: str = "stream",
        kind: DKindT = "positions",
        *,
        ax: Optional[Axes] = None,
        format_ax: bool = False,
        plot_origin: bool = True,
        som: bool = True,
        som_original_prototypes: bool = False,
        som_prototypes_offset: Quantity = Quantity(0.0, u.deg),
        kalman: bool = True,
        kalman_kw: Optional[dict] = None,
    ) -> Axes:
        """Plot Everything.

        Parameters
        ----------
        frame : |Frame| or str, optional
            {frame}
        kind : {DKindT}, optional
            {dkind_ds}

        ax : Optional[Axes], optional
            matplotlib axes, by default `None`.
        format_ax : bool, optional
            Whether to add the axes labels and info, by default `True`.

        som : bool, optional keyword-only
            Whether to plot the SOM, by default `True`.
        som_original_prototypes : bool, optional keyword-only
            Whether to plot the original prototypes, by default `False`.
        som_prototypes_offset : Quantity['angle'], optional keyword-only
            Latitude offset for the SOM prototypes.

        kalman : bool, optional keyword-only
            Whether to plot the Kalman Filter, by default `True`.

        Returns
        -------
        Axes
            The matplotlib axes.
        """
        _, _ax, stream, *_ = self._setup(ax)
        kw = self._get_kw(s=40)
        theframe, frame_name = self._parse_frame(frame)
        xn, yn = self._get_xy_names(theframe, kind=kind)

        # Centrally coordinate plotting the origin since multiple plot methods
        # can plot the origin.
        if plot_origin:
            self.origin(stream.origin, frame=frame, kind=kind, ax=_ax)

        if stream.arm1.has_data:
            c1 = arange(len(stream.arm1))
            stream.arm1.plot.in_frame(
                frame=frame, kind=kind, c=c1, ax=_ax, format_ax=False, plot_origin=False, **kw
            )

        if stream.arm2.has_data:
            c2 = arange(len(stream.arm2.coords_ord))
            stream.arm2.plot.in_frame(
                frame=frame, kind=kind, c=c2, ax=_ax, format_ax=False, plot_origin=False, **kw
            )

        if som:
            self.som(
                frame=frame,
                kind=kind,
                original_prototypes=som_original_prototypes,
                format_ax=False,
                ax=_ax,
                y_offset=som_prototypes_offset,
                plot_origin=False,
            )

        if kalman:
            self.kalman(
                frame=frame,
                kind=kind,
                ax=_ax,
                format_ax=False,
                plot_origin=False,
                **(kalman_kw or {}),
            )

        if format_ax:
            self._format_ax(_ax, frame=frame_name, x=xn, y=yn)

        return _ax

    # ---------------------------------------------------------------

    def fit_frame_multipanel(
        self,
        *,
        axes: Optional[ndarray] = None,
        format_ax: bool = True,
        plot_origin: bool = True,
        **kwargs: Any,
    ) -> Tuple[Figure, ndarray]:
        """Plot frame fit in a 3 pandel plot.

        Parameters
        ----------
        axes : ndarray[Axes] or None, optional
            Maplotlib plot axes on which to plot, by default None
        format_ax : bool, optional
            Whether to add the axes labels and info, by default True
        plot_origin : bool, optional
            Whether to plot the origin, by default True
        **kwargs
            Passed to ``in_icrs_frame``.

        Returns
        -------
        Figure, ndarray[Axes]
            The matplotlib figure and axes.

        Raises
        ------
        Exception
            If the stream does not have a fit frame.
        """
        track, _, stream, full_name, *_ = self._setup(None)
        ORIGIN_HAS_VS = "s" in stream.origin.data.differentials

        _fr = track.frame_fit
        if _fr is None:
            raise Exception("need to fit the stream first")
        fr = cast(FrameOptimizeResult, _fr)

        # Plot setup
        axs: ndarray
        if axes is not None:
            plot_vs = axes.ndim > 1 and axes.shape[1] >= 2 and track.has_kinematics
            axs = axes.reshape(-1, 1) if axes.ndim == 1 else axes
            fig = axs.flat[0].figure  # assuming all the same
        else:
            plot_vs = track.has_kinematics
            nrows = 2 if plot_vs else 1
            figsize = (16 if plot_vs else 8, 7)
            fig, axs = plt.subplots(3, nrows, figsize=figsize)  # type: ignore

        # Plot 1 : Stream in its own frame
        stream.plot.in_frame(
            frame="ICRS",
            kind="positions",
            ax=axs[0, 0],
            format_ax=format_ax,
            plot_origin=plot_origin,
            **kwargs,
        )
        if format_ax:
            axs[0, 0].set_title("Stream Star Positions")
        if plot_vs:
            stream.plot.in_frame(
                frame="ICRS",
                kind="kinematics",
                ax=axs[0, 1],
                format_ax=format_ax,
                plot_origin=plot_origin and ORIGIN_HAS_VS,
                **kwargs,
            )
            if format_ax:
                axs[0, 1].set_title("Stream Star Kinematics")

        # Plot 2 : Residual
        self.fit_frame_residual(ax=axs[1, 0], format_ax=format_ax)
        if plot_vs:
            axs[1, 1].set_axis_off()

        # Plot 3 : Rotated Stream
        sc = stream.coords_ord
        c: Quantity = fr.calculate_residual(sc)
        stream.plot.in_frame(
            frame="stream",
            kind="positions",
            c=c,
            ax=axs[2, 0],
            format_ax=format_ax,
            label=full_name + r" ($\theta=$" f"{fr.rotation.value:.4} [deg])",
            plot_origin=plot_origin,
            **kwargs,
        )
        if plot_vs:
            stream.plot.in_frame(
                frame="stream",
                kind="kinematics",
                c=c,
                ax=axs[2, 1],
                format_ax=format_ax,
                label=full_name + r" ($\theta=$" f"{fr.rotation.value:.4} [deg])",
                plot_origin=plot_origin and ORIGIN_HAS_VS,
                **kwargs,
            )

        return fig, axs

    def som_multipanel(
        self,
        *,
        axes: Optional[Tuple[Axes, Axes]] = None,
        original_prototypes: bool = False,
        plot_origin: bool = True,
        format_ax: bool = True,
        som_prototypes_offset: Quantity = Quantity(0.0, u.deg),
    ) -> Tuple[Figure, Tuple[Axes, Axes]]:
        """Plot SOM applied to the stream.

        Parameters
        ----------
        axes : Optional[Tuple[Axes, Axes]], optional keyword-only
            Maplotlib plot axes on which to plot, by default `None`
        original_prototypes : bool, optional keyword-only
            Whether to plot the original prototypes, by default `False`.
        plot_origin : bool, optional keyword-only
            Whether to plot the origin, by default `True`.
        format_ax : bool, optional
            Whether to add the axes labels and info, by default `True`.
        som_prototypes_offset : Quantity['angle'], optional keyword-only
            Latitude offset for the SOM prototypes.

        Returns
        -------
        Tuple[Figure, Tuple[Axes, Axes]]
            The matplotlib figure and axes.
        """
        _, stream, _, _, *_ = self._setup(None)

        # Plot setup
        if axes is not None:
            ax1, ax2 = axes
            fig = ax1.figure  # assuming all the same
        else:
            fig = plt.figure(figsize=(8, 4))
            ax1 = fig.add_subplot(2, 1, 1)
            ax2 = fig.add_subplot(2, 1, 2)

        # Plot 1 : Stream in ICRS frame
        # Plot 2 : + SOM
        if stream.arm1.has_data:
            c1 = arange(len(stream.arm1))
            stream.arm1.plot.in_frame(
                frame="ICRS",
                kind="positions",
                c=c1,
                ax=ax1,
                plot_origin=plot_origin,
                format_ax=format_ax,
            )

        if stream.arm2.has_data:
            c2 = arange(len(stream.arm2))
            stream.arm2.plot.in_frame(
                frame="ICRS", kind="positions", c=c2, ax=ax1, plot_origin=False, format_ax=False
            )

        self.plot(
            ax=ax2,
            format_ax=format_ax,
            som=True,
            som_original_prototypes=original_prototypes,
            som_prototypes_offset=som_prototypes_offset,
            kalman=False,
        )

        if format_ax:
            fig.tight_layout()

        return fig, (ax1, ax2)

    def full_multipanel(
        self,
        format_ax: bool = True,
        plot_origin: bool = True,
        som_original_prototypes: bool = False,
        som_prototypes_offset: Quantity = Quantity(0.0, u.deg),
        *,
        kalman_kw: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Figure, ndarray]:
        """Plot everything.

        Parameters
        ----------
        format_ax : bool, optional
            Whether to add the axes labels and info, by default True
        plot_origin : bool, optional
            Whether to plot the origin, by default True

        som_original_prototypes : bool, optional
            Whether to plot the original prototypes, by default False
        som_prototypes_offset : Quantity['angle'], optional keyword-only
            Latitude offset for the SOM prototypes.

        Returns
        -------
        Figure, (5, 2) ndarray[Axes]
            The matplotlib figure and axes.
        """
        fig, axs = plt.subplots(5, 2, figsize=(16, 12))
        axs = cast(ndarray, axs)

        track, _, stream, *_ = self._setup(None)

        # Plot frame fit
        self.fit_frame_multipanel(axes=axs[:3, :], format_ax=format_ax, plot_origin=plot_origin)

        # SOM plot
        self.plot(
            ax=axs[3, 0],
            frame="stream",
            kind="positions",
            format_ax=True,
            plot_origin=True,
            som=True,
            som_original_prototypes=som_original_prototypes,
            som_prototypes_offset=som_prototypes_offset,
            kalman=False,
        )
        if track.has_kinematics:
            self.plot(
                ax=axs[3, 1],
                frame="stream",
                kind="kinematics",
                format_ax=True,
                plot_origin="s" in stream.origin.data.differentials,
                som=False,  # SOM not run on kinematics!
                som_original_prototypes=som_original_prototypes,
                som_prototypes_offset=som_prototypes_offset,
                kalman=False,
            )

        # Kalman filter plot
        kalman_kw = {"std": 5} if kalman_kw is None else kalman_kw
        self.plot(
            ax=axs[4, 0],
            frame="stream",
            kind="positions",
            plot_origin=True,
            format_ax=True,
            som=False,
            kalman=True,
            kalman_kw=kalman_kw,
        )
        if track.has_kinematics:
            self.plot(
                ax=axs[4, 1],
                frame="stream",
                kind="kinematics",
                format_ax=True,
                plot_origin="s" in stream.origin.data.differentials,
                som=False,
                kalman=True,
                kalman_kw=kalman_kw,
            )

        if format_ax:
            fig.tight_layout()

        return fig, axs
