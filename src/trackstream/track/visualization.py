"""Stream track plotting."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast, overload

# THIRD PARTY
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import EllipseCollection

# LOCAL
from trackstream.stream.visualization import DKindT, StreamPlotDescriptorBase
from trackstream.track.fit.kalman.core import FirstOrderNewtonianKalmanFilter
from trackstream.track.fit.som import SelfOrganizingMap
from trackstream.track.fit.utils import _v2c
from trackstream.track.utils import covariance_ellipse
from trackstream.utils.visualization import PlotCollectionBase

if TYPE_CHECKING:
    # THIRD PARTY
    from astropy.units import Quantity
    from matplotlib.figure import Figure
    from matplotlib.pyplot import Axes  # type: ignore
    from numpy import ndarray
    from typing_extensions import Unpack

    # LOCAL
    from trackstream._typing import FrameLikeType
    from trackstream.stream.base import StreamBase
    from trackstream.track.core import StreamArmTrack
    from trackstream.track.plural import StreamArmsTrackBase  # noqa: F401


__all__: list[str] = []


##############################################################################
# CODE
##############################################################################


@dataclass
class StreamArmTrackPlotDescriptor(StreamPlotDescriptorBase["StreamArmTrack"]):
    """Plot descriptor for a :class:`trackstream.fit.StreamArmTrack`."""

    @overload
    def _setup(self, *, ax: Axes) -> tuple[StreamArmTrack, Axes, StreamBase, str, Unpack[tuple[Any, ...]]]:
        ...

    @overload
    def _setup(self, *, ax: None) -> tuple[StreamArmTrack, Axes, StreamBase, str, Unpack[tuple[Any, ...]]]:
        ...

    @overload
    def _setup(self, *, ax: Literal[False]) -> tuple[StreamArmTrack, None, StreamBase, str, Unpack[tuple[Any, ...]]]:
        ...

    @overload
    def _setup(self, *, ax: bool) -> tuple[StreamArmTrack, Axes, StreamBase, str, Unpack[tuple[Any, ...]]]:
        ...

    def _setup(
        self, *, ax: Axes | None | bool = None
    ) -> tuple[StreamArmTrack, Axes | None, StreamBase, str, Unpack[tuple[Any, ...]]]:
        """Setup the plot.

        Parameters
        ----------
        ax : |Axes| or None, optional keyword-only
            Matplotlib |Axes|. `None` (default) uses the current axes
            (:func:`matplotlib.pyplot.gca`).

        Returns
        -------
        track : `trackstream.StreamArmTrack`
            The fit stream track.
        ax : |Axes|
            Matplotlib |Axes|.
        stream : `trackstream.Stream`
            The stream which has the `track`.
        full_name : str
            The name of the `track`.
        """
        track, ax, *_ = super()._setup(ax=ax)
        full_name = track.full_name or ""

        stream = track.stream
        if stream is None:
            raise ValueError

        return track, ax, stream, full_name, None

    # ===============================================================
    # Individual plot methods

    def in_frame(
        self,
        frame: str = "stream",
        kind: DKindT = "positions",
        *,
        origin: bool = True,
        ax: Axes | None = None,
        format_ax: bool = False,
        **kwargs: Any,
    ) -> Axes:
        """Plot stream in a stream frame.

        Parameters
        ----------
        frame : |Frame| or str, optional
            A frame instance or its name (a `str`, the default).
            Also supported is "stream", which is the stream frame
            of the enclosing instance.
        kind : {'positions', 'kinematics'}, optional
            The kind of plot.

        origin : bool, optional keyword-only
            Whether to plot the origin, by default `True`.
        ax : |Axes| or None, optional keyword-only
            Matplotlib |Axes|. `None` (default) uses the current axes
            (:func:`matplotlib.pyplot.gca`).
        format_ax : bool, optional keyword-only
            Whether to add the axes labels and info, by default `True`.
        **kwargs : Any
            Keyword arguments passed to :func:`matplotlib.pyplot.scatter`.

        Returns
        -------
        |Axes|
        """
        _, _ax, stream, *_ = self._setup(ax=ax)

        super().in_frame(frame=frame, kind=kind, ax=_ax, format_ax=format_ax, **kwargs)
        # TODO! by arm

        if origin:
            self.origin(frame=frame, kind=kind, ax=_ax)

        return _ax

    def som(
        self,
        frame: FrameLikeType = "stream",
        kind: DKindT = "positions",
        *,
        origin: bool = True,
        connect: bool = True,
        initial_prototypes: bool = False,
        x_offset: u.Quantity | Literal[0] = 0,
        y_offset: u.Quantity | Literal[0] = 0,
        ax: Axes | None = None,
        format_ax: bool = False,
    ) -> Axes:
        """Plot Self-Organizing Map.

        Parameters
        ----------
        frame : |Frame| or str, optional
            A frame instance or its name (a `str`, the default).
            Also supported is "stream", which is the stream frame
            of the enclosing instance.
        kind : {'positions', 'kinematics'}, optional
            The kind of plot.

        origin : bool, optional keyword-only
            Whether to plot the origin, by default `True`.
        initial_prototypes : bool, optional keyword-only
            Whether to plot the initial prototypes, by default `False`.
        y_offset : |Quantity|, optional keyword-only
            Offset in |Latitude| (y coordinate) for the prototypes.

        ax : |Axes| or None, optional keyword-only
            Matplotlib |Axes|. `None` (default) uses the current axes
            (:func:`matplotlib.pyplot.gca`).
        format_ax : bool, optional keyword-only
            Whether to add the axes labels and info, by default `True`.

        Returns
        -------
        |Axes|
        """
        if frame.lower() != "stream":
            raise NotImplementedError

        track, _ax, _, _, *_ = self._setup(ax=ax)
        theframe, frame_name = self._parse_frame(frame)
        som = cast("SelfOrganizingMap", track.som)
        xn, yn = "", ""  # default labels. updated from data.

        if som is not None:
            ps1, _ = self._to_frame(som.prototypes, theframe)
            (x, xn), (y, yn) = self._get_xy(ps1, kind=kind)

            if connect:
                _ax.plot(x + x_offset, y + y_offset, c="k")

            _ax.scatter(
                x + x_offset,
                y + y_offset,
                marker="P",  # type: ignore
                edgecolors="black",
                facecolor="none",
            )

            if initial_prototypes:
                ips, _ = self._to_frame(som.init_prototypes, theframe)
                (x, xn), (y, yn) = self._get_xy(ips, kind=kind)

                _ax.scatter(
                    x,
                    y - y_offset,  # type: ignore
                    marker="X",  # type: ignore
                    edgecolors="gray",
                    facecolor="none",
                )

        if origin:
            self.origin(frame=frame, kind=kind, ax=_ax)

        if format_ax:  # Axes settings
            self._format_ax(_ax, frame=frame_name, x=xn, y=yn)

        return _ax

    # --------------------------------

    def kalman(
        self,
        frame: FrameLikeType = "stream",
        kind: DKindT = "positions",
        *,
        connect: bool = True,
        origin: bool = False,
        ax: Axes | None | bool = None,
        format_ax: bool = False,
        subselect: int = 5,
        **kwargs: Any,
    ) -> Axes:
        """Plot Kalman Filter.

        Parameters
        ----------
        frame : |Frame| or str, optional
            A frame instance or its name (a `str`, the default).
            Also supported is "stream", which is the stream frame
            of the enclosing instance.
        kind : {'positions', 'kinematics'}, optional
            The kind of plot.

        origin : bool, optional keyword-only
            Whether to plot the origin, by default `True`.

        ax : |Axes| or None, optional keyword-only
            Matplotlib |Axes|. `None` (default) uses the current axes
            (:func:`matplotlib.pyplot.gca`).
        format_ax : bool, optional keyword-only
            Whether to add the axes labels and info, by default `True`.

        **kwargs : Any
            Keyword arguments passed to ``_cov``.

        Returns
        -------
        |Axes|
            matplotlib axes.
        """
        if frame.lower() != "stream":
            raise NotImplementedError
        else:
            _, frame_name = self._parse_frame(frame)

        track, _ax, _, _, *_ = self._setup(ax=ax)
        xn, yn = "", ""  # default, updated from data

        kalman = cast("FirstOrderNewtonianKalmanFilter", track.kalman)

        # Subselect from covariance matrix
        start = 0 if kind == "positions" else kalman.nfeature
        slc = (slice(None), slice(start, start + 4, 2), slice(start, start + 4, 2))

        crd = _v2c(kalman, np.array(track.path._meta["smooth"].x[:, ::2]))
        (x, xn), (y, yn) = self._get_xy(crd, kind=kind)
        # error
        Ps = track.path._meta["smooth"].P[slc]

        nstd = kwargs.get("std", 1)
        if connect:
            _ax.plot(x.value, y.value, label=f"track {track.name} : {nstd} std", zorder=100, c=kwargs.get("c"))
        else:
            # still need to auto-set the axes bounds since an EllipseCollection does not.
            _ax.scatter(x.value, y.value, alpha=0)

        # Covariance Ellipses
        if isinstance(subselect, slice):
            subsel = subselect
        else:
            subsel = slice(None, None, subselect)

        mean = np.array((x, y)).reshape((2, -1)).T[subsel]
        angle, wh = covariance_ellipse(Ps[subsel], nstd=nstd)
        width = 2 * np.atleast_1d(wh[..., 0])  # TODO! why 2?
        height = 2 * np.atleast_1d(wh[..., 1])

        ec = EllipseCollection(
            width,
            height,
            angle.to_value(u.deg),
            units="x",
            offsets=np.atleast_2d(mean),
            transOffset=_ax.transData,
            facecolor=kwargs.get("facecolor", "gray"),
            edgecolor=kwargs.get("edgecolor", "none"),
            alpha=kwargs.get("alpha", 0.5),
            lw=2,
            ls=kwargs.get("ls", "solid"),
            zorder=0,
        )
        _ax.add_collection(ec)

        if origin:
            self.origin(frame=frame, kind=kind, ax=_ax)

        if format_ax:
            self._format_ax(_ax, frame=frame_name, x=xn, y=yn)

        return _ax

    # ---------------------------------------------------------------

    def __call__(
        self,
        frame: str = "stream",
        kind: DKindT = "positions",
        *,
        origin: bool = True,
        som: bool = True,
        som_kw: dict | None = None,
        kalman: bool = True,
        kalman_kw: dict | None = None,
        ax: Axes | None = None,
        format_ax: bool = False,
        **kwargs: Any,
    ) -> Axes:
        """Plot anything and everything.

        Parameters
        ----------
        frame : |Frame| or str, optional
            A frame instance or its name (a `str`, the default).
            Also supported is "stream", which is the stream frame
            of the enclosing instance.
        kind : {'positions', 'kinematics'}, optional
            The kind of plot.

        origin : bool, optional keyword-only
            Whether to plot the origin, by default `True`.

        ax : |Axes| or None, optional keyword-only
            Matplotlib |Axes|. `None` (default) uses the current axes
            (:func:`matplotlib.pyplot.gca`).
        format_ax : bool, optional keyword-only
            Whether to add the axes labels and info, by default `True`.

        som : bool, optional keyword-only
            Whether to plot the |SOM|, by default `True`.
        som_kw : dict[str, Any] or None, optional keyword-only
            Keyword arguments passed to ``.som()``.

        kalman : bool, optional keyword-only
            Whether to plot the Kalman Filter, by default `True`.
        kalman_kw : dict[str, Any] or None, optional keyword-only
            Keyword arguments passed to ``.kalman()``.

        **kwargs : Any
            Keyword arguments passed to ``.in_frame()``.

        Returns
        -------
        |Axes|
            The matplotlib figure axes.
        """
        _, _ax, stream, *_ = self._setup(ax=ax)
        kw = self._get_kw(kwargs, c=np.arange(len(stream.coords)))
        theframe, frame_name = self._parse_frame(frame)
        xn, yn = self._get_xy_names(theframe, kind=kind)

        # Centrally coordinate plotting the origin since multiple plot methods
        # can plot the origin.
        if origin:
            self.origin(frame=frame, kind=kind, ax=_ax)

        stream.plot.in_frame(
            frame=frame,
            kind=kind,
            ax=_ax,
            format_ax=False,
            origin=False,
            **kw,
        )

        if som:
            self.som(frame=frame, kind=kind, format_ax=False, ax=_ax, origin=False, **(som_kw or {}))

        if kalman:
            self.kalman(frame=frame, kind=kind, ax=_ax, format_ax=False, origin=False, **(kalman_kw or {}))

        if format_ax:
            self._format_ax(_ax, frame=frame_name, x=xn, y=yn)

        return _ax

    # ---------------------------------------------------------------

    def som_multipanel(
        self,
        *,
        connect: bool = True,
        initial_prototypes: bool = False,
        prototypes_x_offset: u.Quantity | Literal[0] = 0,
        prototypes_y_offset: u.Quantity | Literal[0] = 0,
        origin: bool = True,
        axes: tuple[Axes, Axes] | None = None,
        format_ax: bool = True,
    ) -> tuple[Figure, tuple[Axes, Axes]]:
        """Plot |SOM| applied to the stream.

        Parameters
        ----------
        initial_prototypes : bool, optional keyword-only
            Whether to plot the initial prototypes, by default `False`.
        prototypes_y_offset : Quantity['angle'], optional keyword-only
            |Latitude| offset for the SOM prototypes.
        origin : bool, optional keyword-only
            Whether to plot the origin, by default `True`.

        axes : tuple[|Axes|, |Axes|] or None, optional keyword-only
            Maplotlib plot axes on which to plot, by default `None`
        format_ax : bool, optional keyword-only
            Whether to add the axes labels and info, by default `True`.

        Returns
        -------
        |Figure|
            The matplotlib figure.
        (|Axes|, |Axes|)
            The matplotlib figure axes.
        """
        _, _, stream, _, *_ = self._setup(ax=False)

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
        stream.plot.in_frame(
            frame="icrs",
            kind="positions",
            c=np.arange(len(stream.coords)),
            ax=ax1,
            origin=origin,
            format_ax=format_ax,
        )

        self(
            ax=ax2,
            format_ax=format_ax,
            som=True,
            som_kw=dict(
                connect=connect,
                initial_prototypes=initial_prototypes,
                x_offset=prototypes_x_offset,
                y_offset=prototypes_y_offset,
            ),
            kalman=False,
        )

        if format_ax:
            fig.tight_layout()

        return fig, (ax1, ax2)

    def full_multipanel(
        self,
        *,
        origin: bool = True,
        in_frame_kw: dict[str, Any] | None = None,
        som_kw: dict[str, Any] | None = None,
        kalman_kw: dict[str, Any] | None = None,
        axes: ndarray | None = None,
        format_ax: bool = True,
    ) -> tuple[Figure, ndarray]:
        """Plot everything.

        Parameters
        ----------
        origin : bool, optional keyword-only
            Whether to plot the origin, by default `True`.
        in_frame_kw : dict[str, Any] or None, optional keyword-only
            Options passed to ``.in_frame()``.
        som_kw : dict[str, Any] or None, optional keyword-only
            Options passed to ``.som()``.
        kalman_kw : dict[str, Any] or None, optional keyword-only
            Options passed to ``.kalman()``.

        format_ax : bool, optional keyword-only
            Whether to add the axes labels and info, by default `True`.

        Returns
        -------
        |Figure|
            The matplotlib figure.
        (5, 2) ndarray[|Axes|]
            The matplotlib figure axes.
        """
        track, _, stream, *_ = self._setup(ax=False)

        ORIGIN_HAS_VS = "s" in stream.origin.data.differentials  # type: ignore
        full_name = stream.full_name or ""

        # Plot setup
        axs: ndarray
        if axes is not None:
            plot_vs = axes.ndim > 1 and axes.shape[1] >= 2 and stream.has_kinematics
            axs = axes.reshape(-1, 1) if axes.ndim == 1 else axes
            fig = axs.flat[0].figure  # assuming all the same
        else:
            plot_vs = stream.has_kinematics
            ncols = 2 if plot_vs else 1
            figwidth = 16 if plot_vs else 8
            fig, axs = plt.subplots(3, ncols, figsize=(figwidth, 12))  # type: ignore
            if len(axs.shape) == 1:
                axs.shape = (-1, 1)

        # Plot stream in system frame
        in_frame_kw = {} if in_frame_kw is None else in_frame_kw
        in_frame_kw.setdefault("label", full_name)
        in_frame_kw.setdefault("format_ax", format_ax)
        in_frame_kw.setdefault("origin", origin)
        in_frame_kw.pop("title", True)  # TODO! use
        stream.plot.in_frame(frame="stream", kind="positions", ax=axs[0, 0], **in_frame_kw)
        if plot_vs:
            stream.plot.in_frame(
                frame="stream",
                kind="kinematics",
                ax=axs[0, 1],
                origin=in_frame_kw.pop("origin") and ORIGIN_HAS_VS,
                **in_frame_kw,
            )

        # SOM plot
        som_in_frame_kw = (som_kw or {}).pop("in_frame_kw", {})
        self.__call__(
            ax=axs[1, 0],
            frame="stream",
            kind="positions",
            format_ax=format_ax,
            origin=origin,
            som=True,
            som_kw=som_kw,
            kalman=False,
            **som_in_frame_kw,
        )
        if track.has_kinematics:
            self(
                ax=axs[1, 1],
                frame="stream",
                kind="kinematics",
                format_ax=format_ax,
                origin=origin and "s" in stream.origin.data.differentials,  # type: ignore
                som=True,
                som_kw=som_kw,
                kalman=False,
                **som_in_frame_kw,
            )

        # Kalman filter plot
        kalman_kw = {"std": 3} if kalman_kw is None else kalman_kw
        kalman_in_frame_kw = (kalman_kw or {}).pop("in_frame_kw", {})
        self(
            ax=axs[2, 0],
            frame="stream",
            kind="positions",
            origin=origin,
            format_ax=format_ax,
            som=False,
            kalman=True,
            kalman_kw=kalman_kw,
            **kalman_in_frame_kw,
        )
        if track.has_kinematics:
            self(
                ax=axs[2, 1],
                frame="stream",
                kind="kinematics",
                format_ax=format_ax,
                origin=origin and "s" in stream.origin.data.differentials,  # type: ignore
                som=False,
                kalman=True,
                kalman_kw=kalman_kw,
                **kalman_in_frame_kw,
            )

        if format_ax:
            fig.tight_layout()

        return fig, axs


#####################################################################


@dataclass
class StreamArmsTrackBasePlotDescriptor(PlotCollectionBase["StreamArmsTrackBase"]):
    def in_frame(
        self,
        frame: str = "stream",
        kind: DKindT = "positions",
        *,
        origin: bool = True,
        ax: Axes | None = None,
        format_ax: bool = False,
        **kwargs: Any,
    ) -> dict[str, Axes]:
        track = self.enclosing
        last = len(track.keys()) - 1

        out = {}
        for i, k in enumerate(track.keys()):
            out[k] = track[k].plot.in_frame(
                frame=frame,
                kind=kind,
                origin=False if i != last else origin,
                ax=ax,
                format_ax=False if i != last else format_ax,
                **kwargs,
            )
        return out

    def som(
        self,
        frame: FrameLikeType = "stream",
        kind: DKindT = "positions",
        *,
        origin: bool = True,
        connect: bool = True,
        initial_prototypes: bool = False,
        x_offset: Quantity | Literal[0] = 0,
        y_offset: Quantity | Literal[0] = 0,
        ax: Axes | None = None,
        format_ax: bool = False,
    ) -> dict[str, Axes]:
        track = self.enclosing
        last = len(track.keys()) - 1

        out = {}
        for i, k in enumerate(track.keys()):
            out[k] = track[k].plot.som(
                frame=frame,
                kind=kind,
                connect=connect,
                origin=False if i != last else origin,
                initial_prototypes=initial_prototypes,
                x_offset=x_offset,
                y_offset=y_offset,
                ax=ax,
                format_ax=False if i != last else format_ax,
            )
        return out

    def kalman(
        self,
        frame: FrameLikeType = "stream",
        kind: DKindT = "positions",
        *,
        connect: bool = False,
        origin: bool = False,
        ax: Axes | None = None,
        format_ax: bool = False,
        **kwargs: Any,
    ) -> dict[str, Axes]:
        track = self.enclosing
        last = len(track.keys()) - 1

        out = {}
        for i, k in enumerate(track.keys()):
            out[k] = track[k].plot.kalman(
                frame=frame,
                kind=kind,
                connect=connect,
                origin=False if i != last else origin,
                ax=ax,
                format_ax=False if i != last else format_ax,
                **kwargs,
            )
        return out

    def __call__(
        self,
        frame: str = "stream",
        kind: DKindT = "positions",
        *,
        origin: bool = True,
        som: bool = True,
        som_kw: dict | None = None,
        kalman: bool = True,
        kalman_kw: dict | None = None,
        ax: Axes | None = None,
        format_ax: bool = False,
        **kwargs: Any,
    ) -> dict[str, Axes]:
        track = self.enclosing
        last = len(track.keys()) - 1

        out = {}
        for i, k in enumerate(track.keys()):
            out[k] = track[k].plot(
                frame=frame,
                kind=kind,
                origin=False if i != last else origin,
                som=som,
                som_kw=som_kw,
                kalman=kalman,
                kalman_kw=kalman_kw,
                ax=ax,
                format_ax=False if i != last else format_ax,
                **kwargs,
            )
        return out

    def som_multipanel(
        self,
        *,
        connect: bool = True,
        initial_prototypes: bool = False,
        prototypes_x_offset: Quantity | Literal[0] = 0,
        prototypes_y_offset: Quantity | Literal[0] = 0,
        origin: bool = True,
        axes: tuple[Axes, Axes] | None = None,
        format_ax: bool = True,
    ) -> tuple[Figure | None, tuple[Axes, Axes] | None]:
        track = self.enclosing
        last = len(track.keys()) - 1
        fig: Figure | None = None

        for i, k in enumerate(track.keys()):
            fig, axes = track[k].plot.som_multipanel(
                connect=connect,
                initial_prototypes=initial_prototypes,
                prototypes_x_offset=prototypes_x_offset,
                prototypes_y_offset=prototypes_y_offset,
                origin=False if i != last else origin,
                axes=axes,
                format_ax=False if i != last else format_ax,
            )
        return fig, axes

    def full_multipanel(
        self,
        *,
        origin: bool = True,
        in_frame_kw: dict[str, Any] | None = None,
        som_kw: dict[str, Any] | None = None,
        kalman_kw: dict[str, Any] | None = None,
        axes: ndarray | None = None,
        format_ax: bool = True,
    ) -> tuple[Figure | None, ndarray | None]:
        track = self.enclosing
        last = len(track.keys()) - 1
        fig: Figure | None = None

        in_frame_kw = {} if in_frame_kw is None else in_frame_kw
        som_kw = {} if som_kw is None else som_kw
        kalman_kw = {} if kalman_kw is None else kalman_kw

        for i, k in enumerate(track.keys()):
            ifkw = in_frame_kw.get(k, in_frame_kw)
            skw = som_kw.get(k, som_kw)
            kkw = kalman_kw.get(k, kalman_kw)

            fig, axes = track[k].plot.full_multipanel(
                origin=False if i != last else origin,
                in_frame_kw=ifkw,
                som_kw=skw,
                kalman_kw=kkw,
                axes=axes,
                format_ax=False if i != last else format_ax,
            )
        return fig, axes
