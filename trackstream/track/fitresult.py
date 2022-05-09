# -*- coding: utf-8 -*-

"""Stream track fitter and fit result."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import weakref
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, cast

# THIRD PARTY
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.coordinates import BaseCoordinateFrame, SkyCoord
from astropy.units import Quantity
from astropy.utils.metadata import MetaAttribute, MetaData
from astropy.utils.misc import indent
from interpolated_coordinates import InterpolatedSkyCoord
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from numpy import arange, array, atleast_1d, atleast_2d, ndarray

# LOCAL
from .path import Path, path_moments
from .rotated_frame import FrameOptimizeResult
from trackstream.base import CommonBase
from trackstream.utils.misc import covariance_ellipse
from trackstream.visualization import CLike, StreamPlotDescriptorBase

if TYPE_CHECKING:
    # LOCAL
    from trackstream.stream import Stream  # noqa: E402


__all__ = ["StreamTrack"]


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

    def in_stream_frame(
        self,
        *,
        ax: Optional[Axes] = None,
        c: CLike = "tab:blue",
        format_ax: bool = False,
        **kwargs: Any,
    ) -> Axes:
        """Plot stream in a stream frame.

        Parameters
        ----------
        c : str or array-like[float], optional
            The color or sequence thereof, by default "tab:blue"
        plot_origin : bool, optional
            Whether to plot the origin, by default `True`
        ax : Optional[|Axes|], optional
            Matplotlib |Axes|, by default `None`
        format_ax : bool, optional
            Whether to add the axes labels and info, by default `True`

        Returns
        -------
        |Axes|
        """
        _, _ax, stream, *_ = self._setup(ax)

        super().in_stream_frame(c=c, ax=_ax, format_ax=format_ax, **kwargs)
        # TODO! by arm

        self.origin_label_lonlat(stream.origin.transform_to(stream.frame), ax=_ax)

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

        return fr.plot.residual(stream, ax=_ax, format_ax=format_ax)

    def som(
        self,
        *,
        original_prototypes: bool = False,
        lat_offset: Quantity = Quantity(0.0, u.deg),
        ax: Optional[Axes] = None,
        format_ax: bool = False,
    ) -> Axes:
        """Plot Self-Organizing Map.

        Parameters
        ----------
        original_prototypes : bool, optional keyword-only
            Whether to plot the original prototypes, by default `False`
        lat_offset : Quantity, optional keyword-only
            Offset in latitude (y coordinate) for the prototypes.
        ax : |Axes| or None, optional keyword-only
            Matplotlib |Axes|, by default `None`
        format_ax : bool, optional keyword-only
            Whether to add the axes labels and info, by default `True`

        Returns
        -------
        |Axes|
        """
        track, _ax, stream, _, *_ = self._setup(ax)
        soms = cast(dict, getattr(track, "som", {}))

        # Plot arm 1
        if stream.arm1.has_data and (som1 := soms.get("arm1")) is not None:
            prototypes1: SkyCoord = som1.prototypes
            _ax.scatter(
                prototypes1.lon,
                prototypes1.lat + lat_offset,
                marker="P",  # type: ignore
                edgecolors="black",
                facecolor="none",
            )

            if original_prototypes:
                init_prototypes1: SkyCoord = som1.init_prototypes
                _ax.scatter(
                    init_prototypes1.lon,
                    init_prototypes1.lat - lat_offset,
                    marker="X",  # type: ignore
                    edgecolors="gray",
                    facecolor="none",
                )

        # Plot arm 2
        if stream.arm2.has_data and (som2 := soms.get("arm2")) is not None:
            prototypes2 = som2.prototypes
            _ax.scatter(
                prototypes2.lon,
                prototypes2.lat + lat_offset,
                marker="P",  # type: ignore
                edgecolors="black",
                facecolor="none",
            )

            if original_prototypes:
                init_prototypes2 = som2.init_prototypes
                _ax.scatter(
                    init_prototypes2.lon,
                    init_prototypes2.lat - lat_offset,
                    marker="X",  # type: ignore
                    edgecolors="gray",
                    facecolor="none",
                )

        self.origin_label_lonlat(stream.origin.transform_to(stream.frame), ax=_ax)

        if format_ax:
            _ax.set_xlabel(f"Lon (Stream) [{_ax.get_xlabel()}]", fontsize=13)
            _ax.set_ylabel(f"Lat (Stream) [{_ax.get_ylabel()}]", fontsize=13)
            _ax.grid(True)
            _ax.legend()

        return _ax

    def _cov(
        self,
        ax: Axes,
        mean: ndarray,
        cov: ndarray,
        std: float = 1,
        *,
        facecolor: str = "gray",
        edgecolor: str = "none",
        alpha: float = 1.0,
        ls: str = "solid",
    ) -> None:
        angle, wh = covariance_ellipse(cov)
        width = 2 * atleast_1d(wh[..., 0])
        height = 2 * atleast_1d(wh[..., 0])

        for mn, a, w, h in zip(atleast_2d(mean), angle.to_value(u.deg), width, height):

            e = Ellipse(
                xy=mn,
                width=std * w,
                height=std * h,
                angle=a,
                facecolor=facecolor,
                edgecolor=edgecolor,
                alpha=alpha,
                lw=2,
                ls=ls,
            )
            ax.add_patch(e)

            # if show_center:
            #     x, y = mean
            #     plt.scatter(x, y, marker="+", color=edgecolor)  # type: ignore

    def kalman(self, *, ax: Optional[Axes] = None, format_ax: bool = False, **kwargs) -> Axes:
        """Plot Kalman Filter.

        Parameters
        ----------
        ax : Optional[Axes], optional
            matplotlib axes, by default `None`.

        Returns
        -------
        Axes
            matplotlib axes.
        """
        track, _ax, stream, _, *_ = self._setup(ax)

        # TODO! allow for eval by `affine` using the Path
        # TODO! fill_between 
        # pm = track()
        # ax.fill_between(pm.mean.lon[:, 0], y1=pm.mean.lat[:, 0]-pm.width["lat"]/2, y2=pm.mean.lat[:, 0]+pm.width["lat"]/2)

        if stream.arm1.has_data:
            crd = track.kalman["arm1"]._v_to_crd(array(track.kalman["arm1"]._result.Xs[:, ::2]))
            Ps = track.kalman["arm1"]._result.Ps

            self._cov(
                _ax,
                mean=array((crd.lon, crd.lat)).reshape((2, -1)).T[::5],
                cov=Ps[:, ::2, ::2][::5],
                std=2,
                **kwargs
            )
        if stream.arm2.has_data:
            crd = track.kalman["arm2"]._v_to_crd(array(track.kalman["arm2"]._result.Xs[:, ::2]))
            Ps = track.kalman["arm2"]._result.Ps

            self._cov(
                _ax,
                mean=array((crd.lon, crd.lat)).reshape((2, -1)).T[::5],
                cov=Ps[:, ::2, ::2][::5],
                std=2,
                **kwargs
            )

        if format_ax:
            _ax.set_xlabel(f"Lon (Stream) [{_ax.get_xlabel()}]", fontsize=13)
            _ax.set_ylabel(f"Lat (Stream) [{_ax.get_ylabel()}]", fontsize=13)
            _ax.grid(True)
            _ax.legend()

        return _ax

    # ---------------------------------------------------------------

    def full(
        self,
        *,
        ax: Optional[Axes] = None,
        format_ax: bool = False,
        som: bool = True,
        som_original_prototypes: bool = False,
        som_prototypes_offset: Quantity = Quantity(0.0, u.deg),
        kalman: bool = True,
    ) -> Axes:
        """Plot Everything.

        Parameters
        ----------
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
        _, _ax, stream, _, *_ = self._setup(ax)
        kw = self._get_kw(s=40)

        if stream.arm1.has_data:
            c1 = arange(len(stream.arm1))
            stream.arm1.plot.in_stream_frame(c=c1, ax=_ax, format_ax=False, **kw)

        if stream.arm2.has_data:
            c2 = arange(len(stream.arm2.coords_ord))
            stream.arm2.plot.in_stream_frame(c=c2, ax=_ax, format_ax=False, **kw)

        if som:
            self.som(
                original_prototypes=som_original_prototypes,
                format_ax=False,
                ax=_ax,
                lat_offset=som_prototypes_offset,
            )

        if kalman:
            self.kalman(ax=_ax, format_ax=False)

        if format_ax:
            _ax.set_xlabel(f"Lon (Stream) [{_ax.get_xlabel()}]", fontsize=13)
            _ax.set_ylabel(f"Lat (Stream) [{_ax.get_ylabel()}]", fontsize=13)
            _ax.grid(True)
            _ax.legend()

        return _ax

    # ---------------------------------------------------------------

    def fit_frame_multipanel(
        self,
        *,
        axes: Optional[Tuple[Axes, Axes, Axes]] = None,
        format_ax: bool = True,
        plot_origin: bool = True,
        **kwargs: Any,
    ) -> Tuple[Figure, Tuple[Axes, Axes, Axes]]:
        """Plot frame fit in a 3 pandel plot.

        Parameters
        ----------
        axes : Optional[Tuple[Axes, Axes, Axes]], optional
            Maplotlib plot axes on which to plot, by default None
        format_ax : bool, optional
            Whether to add the axes labels and info, by default True
        plot_origin : bool, optional
            Whether to plot the origin, by default True
        **kwargs
            Passed to ``in_icrs_frame``.

        Returns
        -------
        Figure, (Axes, Axes, Axes)
            The matplotlib figure and axes.

        Raises
        ------
        Exception
            If the stream does not have a fit frame.
        """
        track, _, stream, full_name, *_ = self._setup(None)

        _fr = track.frame_fit
        if _fr is None:
            raise Exception("need to fit the stream first")
        fr = cast(FrameOptimizeResult, _fr)

        # Plot setup
        if axes is not None:
            ax1, ax2, ax3 = axes
            fig = ax1.figure  # assuming all the same
        else:
            fig = plt.figure(figsize=(8, 7))
            ax1 = fig.add_subplot(3, 1, 1)
            ax2 = fig.add_subplot(3, 1, 2)
            ax3 = fig.add_subplot(3, 1, 3)

        # Plot 1 : Stream in its own frame
        stream.plot.in_icrs_frame(ax=ax1, format_ax=format_ax, **kwargs)

        # Plot 2 : Residual
        self.fit_frame_residual(ax=ax2, format_ax=format_ax)

        # Plot 3 : Rotated Stream
        sc = stream.coords_ord
        c: Quantity = fr.calculate_residual(sc)
        stream.plot.in_stream_frame(
            c=c,
            ax=ax3,
            format_ax=format_ax,
            label=full_name + r" ($\theta=$" f"{fr.rotation.value:.4} [deg])",
            plot_origin=plot_origin,
            **kwargs,
        )

        return fig, (ax1, ax2, ax3)

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
            stream.arm1.plot.in_icrs_frame(
                c=c1, ax=ax1, plot_origin=plot_origin, format_ax=format_ax
            )

        if stream.arm2.has_data:
            c2 = arange(len(stream.arm2))
            stream.arm2.plot.in_icrs_frame(c=c2, ax=ax1, plot_origin=False, format_ax=False)

        self.full(
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
    ) -> Tuple[Figure, Tuple[Axes, Axes, Axes, Axes, Axes]]:
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
        Figure, (Axes, Axes, Axes, Axes, Axes)
            The matplotlib figure and axes.
        """
        fig = plt.figure(figsize=(8, 12))
        ax1 = fig.add_subplot(5, 1, 1)
        ax2 = fig.add_subplot(5, 1, 2)
        ax3 = fig.add_subplot(5, 1, 3)
        ax4 = fig.add_subplot(5, 1, 4)
        ax5 = fig.add_subplot(5, 1, 5)

        # Plot frame fit
        self.fit_frame_multipanel(
            axes=(ax1, ax2, ax3), format_ax=format_ax, plot_origin=plot_origin
        )

        # SOM plot
        self.full(
            ax=ax4,
            format_ax=False,
            som=True,
            som_original_prototypes=som_original_prototypes,
            som_prototypes_offset=som_prototypes_offset,
            kalman=False,
        )

        # Kalman filter plot
        self.full(ax=ax5, format_ax=False, som=False, kalman=True)
        # self.kalman(ax=ax5, format_ax=format_ax)

        if format_ax:
            fig.tight_layout()

        return fig, (ax1, ax2, ax3, ax4, ax5)


class StreamTrack(CommonBase):
    """A stream track interpolation as function of arc length.

    The track is Callable, returning a Frame.

    Parameters
    ----------
    stream : `~trackstream.stream.Stream` path : `~trackstream.utils.path.Path`
    origin : SkyCoord
        of the coordinate system (often the progenitor)

    name : str or None, optional keyword-only **meta : Any
        Metadata. Can include the meta-attributes ``frame_fit``,
        ``visit_order``, ``som``, and ``kalman``.
    """

    _name: Optional[str]
    _meta: Dict[str, Any]
    meta = MetaData()

    frame_fit = MetaAttribute()
    visit_order = MetaAttribute()
    som = MetaAttribute()
    kalman = MetaAttribute()

    plot = StreamTrackPlotDescriptor()

    def __init__(
        self,
        stream: "Stream",
        path: Path,
        origin: SkyCoord,
        *,
        name: Optional[str] = None,
        **meta: Any,
    ) -> None:
        super().__init__(frame=path.frame, representation_type=None, differential_type=None)
        self._name = name
        self._stream_ref = weakref.ref(stream)  # reference to original stream

        # validation of types
        if not isinstance(path, Path):
            raise TypeError("`path` must be <Path>.")
        elif not isinstance(origin, (SkyCoord, BaseCoordinateFrame)):
            raise TypeError("`origin` must be <|SkyCoord|, |Frame|>.")

        # assign
        self._path: Path = path
        self._origin = origin

        # set the MetaAttribute(s)
        for attr in list(meta):
            descr = getattr(self.__class__, attr, None)
            if isinstance(descr, MetaAttribute):
                setattr(self, attr, meta.pop(attr))
        # and the meta
        self._meta.update(meta)

    @property
    def stream(self) -> Optional["Stream"]:
        """The `~trackstream.Stream`, or `None` if the weak reference is broken."""
        return self._stream_ref()

    @property
    def name(self) -> Optional[str]:
        """Return the stream-track name."""
        return self._name

    @property
    def full_name(self) -> Optional[str]:
        return self._name

    @property
    def path(self) -> Path:
        return self._path

    @property
    def coords(self) -> InterpolatedSkyCoord:
        """The path's central track."""
        return self._path.data

    @property
    def affine(self) -> Quantity:
        return self._path.affine

    @property
    def origin(self) -> SkyCoord:
        return self._origin

    @property
    def frame(self) -> BaseCoordinateFrame:
        crds = self.coords
        frame = crds.frame.replicate_without_data()
        frame.representation_type = crds.representation_type
        frame.differential_type = crds.differential_type
        return frame

    #######################################################
    # Math on the Track

    def __call__(self, affine: Optional[Quantity] = None, *, angular: bool = False) -> path_moments:
        """Get discrete points along interpolated stream track.

        Parameters
        ----------
        affine : `~astropy.units.Quantity` array-like or None, optional
            The affine interpolation parameter. If None (default), return
            path moments evaluated at all "tick" interpolation points.
        angular : bool, optional keyword-only
            Whether to compute on-sky or real-space.

        Returns
        -------
        `trackstream.utils.path.path_moments`
            Realized from the ``.path`` attribute.
        """
        # TODO! add amplitude (density)
        return self.path(affine=affine, angular=angular)

    def probability(
        self,
        point: SkyCoord,
        background_model: Optional[Callable[[SkyCoord], Quantity[u.percent]]] = None,
        *,
        angular: bool = False,
        affine: Optional[Quantity] = None,
    ) -> Quantity[u.percent]:
        """Probability point is part of the stream.

        .. todo:: angular probability

        """
        # # Background probability
        # Pb = background_model(point) if background_model is not None else 0.0

        # angular = False  # TODO: angular probability
        # afn = self._path.closest_affine_to_point(point, angular=False, affine=affine)
        # pt_w = getattr(self._path, "width_angular" if angular else "width")(afn)
        # sep = getattr(self._path, "separation" if angular else "separation_3d")(
        #     point,
        #     interpolate=False,
        #     affine=afn,
        # )

        # # cov = 1  # Assumption
        # pdf = exp(-0.5 * sep ** 2) / power(2 * pi, 3.0 / 2)
        # # TODO! multidimensional PDF

        raise NotImplementedError

    #######################################################
    # misc

    def __repr__(self) -> str:
        """String representation."""
        rs = []

        # 0) header (standard repr)
        header: str = object.__repr__(self)
        frame_name = self.frame.__class__.__name__
        rep_name = self.coords.representation_type.__name__
        header = header.replace("StreamTrack", f"StreamTrack ({frame_name}|{rep_name})")
        rs.append(header)

        # 1) name
        name = str(self.name)
        rs.append("  Name: " + name)

        # 2) data
        rs.append(indent(repr(self._path.data), width=2))

        return "\n".join(rs)
