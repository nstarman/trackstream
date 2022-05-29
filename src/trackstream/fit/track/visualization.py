# -*- coding: utf-8 -*-

"""Stream track plotting."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    cast,
    overload,
)

# THIRD PARTY
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.units import Quantity
from attrs import define
from matplotlib.collections import EllipseCollection
from matplotlib.figure import Figure
from matplotlib.pyplot import Axes
from numpy import arange, array, atleast_1d, atleast_2d, ndarray
from typing_extensions import Unpack

# LOCAL
from trackstream._type_hints import FrameLikeType
from trackstream.fit.kalman import FirstOrderNewtonianKalmanFilter
from trackstream.fit.som import SelfOrganizingMap1DBase
from trackstream.utils.misc import covariance_ellipse
from trackstream.visualization import (
    DKindT,
    PlotCollectionBase,
    StreamPlotDescriptorBase,
)

if TYPE_CHECKING:
    # LOCAL
    from trackstream.fit.track import StreamArmTrack
    from trackstream.fit.track.plural import StreamArmsTrackBase  # noqa: F401
    from trackstream.stream.base import StreamBase


__all__: List[str] = []


##############################################################################
# CODE
##############################################################################


class StreamArmTrackPlotDescriptor(StreamPlotDescriptorBase["StreamArmTrack"]):
    """Plot descriptor for a :class:`trackstream.fit.StreamArmTrack`."""

    @overload
    def _setup(
        self, *, ax: Axes
    ) -> Tuple["StreamArmTrack", Axes, "StreamBase", str, Unpack[Tuple[Any, ...]]]:
        ...

    @overload
    def _setup(
        self, *, ax: None
    ) -> Tuple["StreamArmTrack", Axes, "StreamBase", str, Unpack[Tuple[Any, ...]]]:
        ...

    @overload
    def _setup(
        self, *, ax: Literal[False]
    ) -> Tuple["StreamArmTrack", None, "StreamBase", str, Unpack[Tuple[Any, ...]]]:
        ...

    def _setup(
        self, *, ax: Union[Axes, None, Literal[False]] = None
    ) -> Tuple["StreamArmTrack", Optional[Axes], "StreamBase", str, Unpack[Tuple[Any, ...]]]:
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
        ax: Optional[Axes] = None,
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
            self.origin(stream.origin, frame=frame, kind=kind, ax=_ax)

        return _ax

    def som(
        self,
        frame: FrameLikeType = "stream",
        kind: DKindT = "positions",
        *,
        origin: bool = True,
        initial_prototypes: bool = False,
        y_offset: Quantity = Quantity(0.0, u.deg),
        ax: Optional[Axes] = None,
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
        elif kind != "positions":
            raise ValueError("SOM is only run on positions")

        track, _ax, stream, _, *_ = self._setup(ax=ax)
        theframe, frame_name = self._parse_frame(frame)
        som = track.som
        xn, yn = "", ""  # default labels. updated from data.

        # Plot arm 1
        if som is not None:
            som = cast(SelfOrganizingMap1DBase, som)
            ps1, _ = self._to_frame(som.prototypes_crd, theframe)
            (x, xn), (y, yn) = self._get_xy(ps1, kind=kind)

            _ax.scatter(
                x,
                y + y_offset,
                marker="P",  # type: ignore
                edgecolors="black",
                facecolor="none",
            )

            if initial_prototypes:
                ips, _ = self._to_frame(som.init_prototypes_crd, theframe)
                (x, xn), (y, yn) = self._get_xy(ips, kind=kind)

                _ax.scatter(
                    x,
                    y - y_offset,  # type: ignore
                    marker="X",  # type: ignore
                    edgecolors="gray",
                    facecolor="none",
                )

        if origin:
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
        """Plot covariance |Ellipse|.

        Plotted as a collection using
        :class:`matplotlib.collections.EllipseCollection`.

        Parameters
        ----------
        ax : |Axes|
            Matplotlib axes onto which to plot the covariance |Ellipse|.
        mean : (N?, 2) ndarray
            Rows are the central positions of the covariance |Ellipse|.
        cov : (N?, 2, 2) ndarray
            N x (2, 2) covariance matrices.
        std : float, optional
            Number of standard deviations of the covariance |Ellipse| axis,
            by default 1.

        facecolor : str, optional keyword-only
            The facecolor of the |Ellipse|, by default 'gray'.
        edgecolor : str, optional keyword-only
            The edgecolor of the |Ellipse|, by default 'none'.
        alpha : float, optional keyword-only
            Transparency, by default 0.5.
        ls : str, optional keyword-only
            Line style, by default solid.
        """
        angle, wh = covariance_ellipse(cov, nstd=1)
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

    def kalman(
        self,
        frame: FrameLikeType = "stream",
        kind: DKindT = "positions",
        *,
        origin: bool = False,
        ax: Optional[Axes] = None,
        format_ax: bool = False,
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

        track, _ax, stream, _, *_ = self._setup(ax=ax)
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

        kalman = cast(FirstOrderNewtonianKalmanFilter, track.kalman)
        crd = kalman._v_to_crd(array(track.path._meta["smooth"].Xs[:, ::2]))
        (x, xn), (y, yn) = self._get_xy(crd, kind=kind)
        # error
        Ps = track.path._meta["smooth"].Ps[slc]

        self._cov(_ax, mean=array((x, y)).reshape((2, -1)).T[::5], cov=Ps[::5], **kwargs)

        if origin:
            self.origin(stream.origin, frame=frame, kind=kind, ax=_ax)

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
        som_initial_prototypes: bool = False,
        som_prototypes_offset: Quantity = Quantity(0.0, u.deg),
        kalman: bool = True,
        kalman_kw: Optional[dict] = None,
        ax: Optional[Axes] = None,
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
        som_initial_prototypes : bool, optional keyword-only
            Whether to plot the initial prototypes, by default `False`.
        som_prototypes_offset : Quantity['angle'], optional keyword-only
            |Latitude| offset for the SOM prototypes.

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
        kw = self._get_kw(kwargs)
        theframe, frame_name = self._parse_frame(frame)
        xn, yn = self._get_xy_names(theframe, kind=kind)

        # Centrally coordinate plotting the origin since multiple plot methods
        # can plot the origin.
        if origin:
            self.origin(stream.origin, frame=frame, kind=kind, ax=_ax)

        stream.plot.in_frame(
            frame=frame,
            kind=kind,
            c=arange(len(stream)),
            ax=_ax,
            format_ax=False,
            origin=False,
            **kw,
        )

        if som:
            self.som(
                frame=frame,
                kind=kind,
                initial_prototypes=som_initial_prototypes,
                format_ax=False,
                ax=_ax,
                y_offset=som_prototypes_offset,
                origin=False,
            )

        if kalman:
            self.kalman(
                frame=frame,
                kind=kind,
                ax=_ax,
                format_ax=False,
                origin=False,
                **(kalman_kw or {}),
            )

        if format_ax:
            self._format_ax(_ax, frame=frame_name, x=xn, y=yn)

        return _ax

    # ---------------------------------------------------------------

    def som_multipanel(
        self,
        *,
        initial_prototypes: bool = False,
        prototypes_offset: Quantity = Quantity(0.0, u.deg),
        origin: bool = True,
        axes: Optional[Tuple[Axes, Axes]] = None,
        format_ax: bool = True,
    ) -> Tuple[Figure, Tuple[Axes, Axes]]:
        """Plot |SOM| applied to the stream.

        Parameters
        ----------
        initial_prototypes : bool, optional keyword-only
            Whether to plot the initial prototypes, by default `False`.
        prototypes_offset : Quantity['angle'], optional keyword-only
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
        _, _, stream, _, *_ = self._setup(ax=None)

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
            frame="ICRS",
            kind="positions",
            c=arange(len(stream)),
            ax=ax1,
            origin=origin,
            format_ax=format_ax,
        )

        self(
            ax=ax2,
            format_ax=format_ax,
            som=True,
            som_initial_prototypes=initial_prototypes,
            som_prototypes_offset=prototypes_offset,
            kalman=False,
        )

        if format_ax:
            fig.tight_layout()

        return fig, (ax1, ax2)

    def full_multipanel(
        self,
        *,
        origin: bool = True,
        som_initial_prototypes: bool = False,
        som_prototypes_offset: Quantity = Quantity(0.0, u.deg),
        kalman_kw: Optional[Dict[str, Any]] = None,
        axes: Optional[ndarray] = None,
        format_ax: bool = True,
    ) -> Tuple[Figure, ndarray]:
        """Plot everything.

        Parameters
        ----------
        origin : bool, optional keyword-only
            Whether to plot the origin, by default `True`.
        som_initial_prototypes : bool, optional
            Whether to plot the original prototypes, by default False
        som_prototypes_offset : Quantity['angle'], optional keyword-only
            |Latitude| offset for the SOM prototypes.
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
            ncols = 2 if plot_vs else 1
            figwidth = 16 if plot_vs else 8
            fig, _axs = plt.subplots(3, ncols, figsize=(figwidth, 12))
            axs = cast(ndarray, _axs)
            if len(axs.shape) == 1:
                axs.shape = (-1, 1)

        # Plot stream in system frame
        stream.plot.in_frame(
            frame="stream",
            kind="positions",
            ax=axs[0, 0],
            format_ax=format_ax,
            label=full_name,
            origin=origin,
        )
        if plot_vs:
            stream.plot.in_frame(
                frame="stream",
                kind="kinematics",
                ax=axs[0, 1],
                format_ax=format_ax,
                label=full_name,
                origin=origin and ORIGIN_HAS_VS,
            )

        # SOM plot
        self(
            ax=axs[1, 0],
            frame="stream",
            kind="positions",
            format_ax=format_ax,
            origin=origin,
            som=True,
            som_initial_prototypes=som_initial_prototypes,
            som_prototypes_offset=som_prototypes_offset,
            kalman=False,
        )
        if track.has_kinematics:
            self(
                ax=axs[1, 1],
                frame="stream",
                kind="kinematics",
                format_ax=format_ax,
                origin=origin and "s" in stream.origin.data.differentials,
                som=False,  # SOM not run on kinematics!
                som_initial_prototypes=som_initial_prototypes,
                som_prototypes_offset=som_prototypes_offset,
                kalman=False,
            )

        # Kalman filter plot
        kalman_kw = {"std": 5} if kalman_kw is None else kalman_kw
        self(
            ax=axs[2, 0],
            frame="stream",
            kind="positions",
            origin=origin,
            format_ax=format_ax,
            som=False,
            kalman=True,
            kalman_kw=kalman_kw,
        )
        if track.has_kinematics:
            self(
                ax=axs[2, 1],
                frame="stream",
                kind="kinematics",
                format_ax=format_ax,
                origin=origin and "s" in stream.origin.data.differentials,
                som=False,
                kalman=True,
                kalman_kw=kalman_kw,
            )

        if format_ax:
            fig.tight_layout()

        return fig, axs


#####################################################################


@define(frozen=True, init=False)
class StreamArmsTrackBasePlotDescriptor(PlotCollectionBase["StreamArmsTrackBase"]):
    def in_frame(
        self,
        frame: str = "stream",
        kind: DKindT = "positions",
        *,
        origin: bool = True,
        ax: Optional[Axes] = None,
        format_ax: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Axes]:
        track = self._enclosing
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
        initial_prototypes: bool = False,
        y_offset: Quantity = Quantity(0.0, u.deg),
        ax: Optional[Axes] = None,
        format_ax: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Axes]:
        track = self._enclosing
        last = len(track.keys()) - 1

        out = {}
        for i, k in enumerate(track.keys()):
            out[k] = track[k].plot.som(
                frame=frame,
                kind=kind,
                origin=False if i != last else origin,
                initial_prototypes=initial_prototypes,
                y_offset=y_offset,
                ax=ax,
                format_ax=False if i != last else format_ax,
                **kwargs,
            )
        return out

    def kalman(
        self,
        frame: FrameLikeType = "stream",
        kind: DKindT = "positions",
        *,
        origin: bool = False,
        ax: Optional[Axes] = None,
        format_ax: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Axes]:
        track = self._enclosing
        last = len(track.keys()) - 1

        out = {}
        for i, k in enumerate(track.keys()):
            out[k] = track[k].plot.kalman(
                frame=frame,
                kind=kind,
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
        som_initial_prototypes: bool = False,
        som_prototypes_offset: Quantity = Quantity(0.0, u.deg),
        kalman: bool = True,
        kalman_kw: Optional[dict] = None,
        ax: Optional[Axes] = None,
        format_ax: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Axes]:
        track = self._enclosing
        last = len(track.keys()) - 1

        out = {}
        for i, k in enumerate(track.keys()):
            out[k] = track[k].plot.kalman(
                frame=frame,
                kind=kind,
                origin=False if i != last else origin,
                som=som,
                som_initial_prototypes=som_initial_prototypes,
                som_prototypes_offset=som_prototypes_offset,
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
        initial_prototypes: bool = False,
        prototypes_offset: Quantity = Quantity(0.0, u.deg),
        origin: bool = True,
        axes: Optional[Tuple[Axes, Axes]] = None,
        format_ax: bool = True,
    ) -> Tuple[Optional[Figure], Optional[Tuple[Axes, Axes]]]:
        track = self._enclosing
        last = len(track.keys()) - 1
        fig: Optional[Figure] = None

        for i, k in enumerate(track.keys()):
            fig, axes = track[k].plot.som_multipanel(
                initial_prototypes=initial_prototypes,
                prototypes_offset=prototypes_offset,
                origin=False if i != last else origin,
                axes=axes,
                format_ax=False if i != last else format_ax,
            )
        return fig, axes

    def full_multipanel(
        self,
        *,
        origin: bool = True,
        som_initial_prototypes: bool = False,
        som_prototypes_offset: Quantity = Quantity(0.0, u.deg),
        kalman_kw: Optional[Dict[str, Any]] = None,
        axes: Optional[ndarray] = None,
        format_ax: bool = True,
    ) -> Tuple[Optional[Figure], Optional[ndarray]]:
        track = self._enclosing
        last = len(track.keys()) - 1
        fig: Optional[Figure] = None

        for i, k in enumerate(track.keys()):
            fig, axes = track[k].plot.full_multipanel(
                origin=False if i != last else origin,
                som_initial_prototypes=som_initial_prototypes,
                som_prototypes_offset=som_prototypes_offset,
                kalman_kw=kalman_kw,
                axes=axes,
                format_ax=False if i != last else format_ax,
            )
        return fig, axes
