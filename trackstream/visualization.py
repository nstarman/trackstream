# -*- coding: utf-8 -*-

"""Utilities for :mod:`~trackstream.utils`."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from typing import TYPE_CHECKING, Any, Optional, Protocol, Sequence, Tuple, TypeVar, Union, cast

# THIRD PARTY
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import ICRS, Angle, BaseCoordinateFrame, CartesianRepresentation
from astropy.coordinates import Longitude, SkyCoord, SphericalRepresentation
from astropy.units import Quantity
from astropy.visualization import imshow_norm
from matplotlib.figure import Figure
from matplotlib.patheffects import withStroke
from matplotlib.pyplot import Axes
from numpy import array, ndarray

# LOCAL
from trackstream.utils.descriptors import InstanceDescriptor

# This is to solve the circular dependency in type hint forward references # isort: skip
if TYPE_CHECKING:
    # LOCAL
    from trackstream.stream import Stream  # noqa: E402

##############################################################################
# PARAMETERS

CLike = Union[str, Sequence[float], Quantity]


##############################################################################
# CODE
##############################################################################


def plot_rotation_frame_residual(
    stream: "Stream", num_rots: int = 3600, scalar: bool = True, **kwargs: Any
) -> Tuple[Figure, Axes]:
    """Plot residual from finding the optimal rotated frame.

    Parameters
    ----------
    stream : `trackstream.stream.Stream`
    num_rots : int, optional
        Number of rotation angles in (-180, 180) to plot.
    scalar : bool, optional
        Whether to plot scalar or full vector residual.

    Returns
    -------
    `~matplotlib.pyplot.Figure`
    """
    # LOCAL
    from .rotated_frame import residual

    # Get data
    frame = stream.frame
    origin = stream.origin.transform_to(frame).represent_as(SphericalRepresentation)
    lon = origin.lon.to_value(u.deg)
    lat = origin.lat.to_value(u.deg)

    # Evaluate residual
    rotation_angles = np.linspace(-180, 180, num=num_rots)
    res = np.array(
        [
            residual(
                (angle, lon, lat),
                data=stream.coords.represent_as(CartesianRepresentation),
                scalar=scalar,
            )
            for angle in rotation_angles
        ],
    )

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    if scalar:
        ax.scatter(rotation_angles, res, **kwargs)
        ax.set_xlabel(r"Rotation angle $\theta$")
        ax.set_ylabel("residual")

    else:
        im, norm = imshow_norm(res, ax=ax, aspect="auto", origin="lower", **kwargs)
        # yticks
        ylocs = ax.get_yticks()
        yticks = [str(int(loc * 360 / num_rots) - 180) for loc in ylocs]
        ax.set_yticks(ylocs[1:-1], yticks[1:-1])
        # labels
        ax.set_xlabel(r"data index")
        ax.set_ylabel(r"Rotation angle $\theta$ [deg]")

        # colorbar
        cbar = fig.colorbar(im)
        cbar.ax.set_ylabel("residual")

    return fig, ax


# -------------------------------------------------------------------


def plot_SOM(data: ndarray, order: ndarray) -> Figure:
    """Plot SOM.

    Parameters
    ----------
    data
    order

    returns

    """
    fig, ax = plt.subplots(figsize=(10, 9))

    pts = ax.scatter(
        data[order, 0],
        data[order, 1],
        c=np.arange(0, len(data)),
        vmax=len(data),
        cmap=plt.get_cmap("plasma"),
        label="data",
    )

    ax.plot(data[order][:, 0], data[order][:, 1], c="gray")

    cbar = plt.colorbar(pts, ax=ax)
    cbar.ax.set_ylabel("SOM ordering")

    fig.legend(loc="upper left")
    fig.tight_layout()

    return fig


##############################################################################


class NamedWithCoords(Protocol):
    @property
    def full_name(self) -> Optional[str]:
        ...

    @property
    def coords(self) -> SkyCoord:
        ...

    @property
    def frame(self) -> BaseCoordinateFrame:
        ...


StreamLikeType = TypeVar("StreamLikeType", bound=NamedWithCoords)


class StreamPlotDescriptorBase(InstanceDescriptor[StreamLikeType]):
    def __init__(self) -> None:
        super().__init__()

        self._default_scatter_style = {"marker": "*", "s": 3}

    def _plot_setup(self, ax: Optional[plt.Axes]) -> Tuple[StreamLikeType, plt.Axes]:
        # Stream
        parent = self._parent
        # Plot axes
        _ax = ax if ax is not None else plt.gca()  # get Axes instance

        return parent, _ax

    def _wrap_stream_lon_order(
        self,
        lon: Angle,
        cut_at: Angle = Angle(100, u.deg),
        wrap_by: Angle = Angle(-360, u.deg),
    ) -> Tuple[Angle, np.ndarray]:

        lt = np.where(lon < cut_at)[0]
        gt = np.where(lon > cut_at)[0]

        order = np.concatenate((gt, lt))
        lon = np.concatenate((lon[gt] + wrap_by, lon[lt]))

        return lon, order

    # ===============================================================

    def in_icrs_frame(
        self,
        c: CLike = "tab:blue",
        *,
        ax: Optional[plt.Axes] = None,
        format_ax: bool = True,
        **kwargs: Any,
    ) -> plt.Axes:
        stream, _ax = self._plot_setup(ax)
        icrs = stream.coords.frame.transform_to(ICRS())  # TODO! report bug in astropy

        kwargs.setdefault("label", stream.full_name)
        kw = {**self._default_scatter_style, **kwargs}
        _ax.scatter(icrs.ra.wrap_at(Angle(180, u.deg)), icrs.dec, c=c, **kw)  # type: ignore

        if format_ax:
            _ax.set_xlabel(f"RA (ICRS) [{_ax.get_xlabel()}]", fontsize=13)
            _ax.set_ylabel(f"Dec (ICRS) [{_ax.get_ylabel()}]", fontsize=13)
            _ax.grid(True)
            _ax.legend()

        return _ax

    def in_stream_frame(
        self,
        c: CLike = "tab:blue",
        *,
        ax: Optional[plt.Axes] = None,
        format_ax: bool = True,
        **kwargs: Any,
    ) -> plt.Axes:
        stream, _ax = self._plot_setup(ax)

        # # Horizontal line on the origin
        # _ax.axhline(0, c="gray", ls="--", zorder=0, alpha=0.5)

        # Plot Stream
        kwargs.setdefault("label", stream.full_name)
        kw = {**self._default_scatter_style, **kwargs}
        sc = stream.coords.transform_to(stream.frame)
        lon, sorter = self._wrap_stream_lon_order(sc.lon, Angle(100, u.deg), Angle(-360, u.deg))
        if not isinstance(c, str):
            c = array(c, copy=False)[sorter]

        _ax.scatter(lon, sc.lat[sorter], c=c, **kw)  # type: ignore

        # Axes settings
        if format_ax:
            _ax.set_xlabel(f"Lon (Stream) [{_ax.get_xlabel()}]", fontsize=13)
            _ax.set_ylabel(f"Lat (Stream) [{_ax.get_ylabel()}]", fontsize=13)
            _ax.grid(True)
            _ax.legend()

        return _ax

    # ========================================================================

    def plot_origin_label_lonlat(
        self, lon: Longitude, lat: Quantity, text_offset: float = -7, *, ax: Optional[plt.Axes]
    ) -> plt.Axes:
        # TODO! automate
        _, _ax = self._plot_setup(ax)

        x = cast(Longitude, lon.wrap_at(Angle(180, u.deg)))
        y: Quantity = lat

        # Plot the central point
        _ax.scatter(x.to_value(u.deg), y.to_value(u.deg), s=10, color="red")
        # Add surrounding circle
        # circle = plt.Circle(
        #     (lon.wrap_at(180 * u.deg).value, lat.value),
        #     4,
        #     clip_on=False,
        #     zorder=10,
        #     linewidth=2.0,
        #     edgecolor="red",
        #     facecolor="none",
        #     path_effects=[withStroke(linewidth=7, foreground=(1, 1, 1, 1))],
        #     # transform=_ax.transAxes,
        # )
        # _ax.add_artist(circle)
        _ax.scatter(x.to_value(u.deg), y.to_value(u.deg), s=1000, facecolor="None", edgecolor="red")

        # Add text
        _ax.text(
            x.value,
            y.value + text_offset,
            "origin",
            zorder=100,
            ha="center",
            va="center",
            weight="bold",
            color="red",
            style="italic",
            fontfamily="monospace",
            path_effects=[withStroke(linewidth=7, foreground=(1, 1, 1, 1))],
        )

        return _ax
