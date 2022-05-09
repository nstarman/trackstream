# -*- coding: utf-8 -*-

"""Utilities for :mod:`~trackstream.utils`."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from typing import Any, Dict, Optional, Protocol, Sequence, Tuple, TypeVar, Union, cast
from typing import runtime_checkable

# THIRD PARTY
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import ICRS, Angle, BaseCoordinateFrame, Longitude, SkyCoord
from astropy.coordinates import SphericalRepresentation
from astropy.units import Quantity
from astropy.visualization import quantity_support
from matplotlib.pyplot import Axes

# LOCAL
from trackstream._type_hints import CoordinateType
from trackstream.utils.descriptors import EnclType, InstanceDescriptor

##############################################################################
# PARAMETERS

CLike = Union[str, Sequence[float], Quantity]

# Ensure Quantity is supported in plots
quantity_support()

##############################################################################
# CODE
##############################################################################


class PlotDescriptorBase(InstanceDescriptor[EnclType]):
    """Plot descriptor base class.

    Parameters
    ----------
    default_scatter_style: dict[str, Any] or None, optional
    """

    _default_scatter_style: Dict[str, Any]

    def __init__(self, *, default_scatter_style: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()

        # Default scatter style
        scatter_style = default_scatter_style or {}
        scatter_style.setdefault("marker", "*")
        scatter_style.setdefault("s", 3)
        self._default_scatter_style = scatter_style

    def _get_kw(self, kwargs: Optional[Dict[str, Any]] = None, **defaults: Any) -> Dict[str, Any]:
        """Get plot options.

        Parameters
        ----------
        kwargs : dict[str, Any]
            Plot options.
        **defaults: Any
            Default plot options

        Returns
        -------
        dict[str, Any]
            Mix of ``kwargs``, ``defaults``, and ``_default_scatter_style``,
            preferring them in that order.
        """
        kw: Dict[str, Any] = {**self._default_scatter_style, **defaults, **(kwargs or {})}
        return kw

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
        # Stream
        parent = self._enclosing
        # Plot axes
        _ax = ax if ax is not None else plt.gca()  # get Axes instance

        return parent, _ax


##############################################################################


@runtime_checkable
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


class StreamPlotDescriptorBase(PlotDescriptorBase[StreamLikeType]):
    """Plot descriptor base class.

    Parameters
    ----------
    default_scatter_style: dict[str, Any] or None, optional
    """

    def _wrap_lon_order(
        self,
        lon: Angle,
        cut_at: Angle = Angle(100, u.deg),
        wrap_by: Angle = Angle(-360, u.deg),
    ) -> Tuple[Angle, np.ndarray]:
        """Wrap the stream by `~astropy.coordinates.Longitude`.

        Parameters
        ----------
        lon : Angle
            Longitude.
        cut_at : Angle, optional
            Angle at which to cut, by default Angle(100, u.deg)
        wrap_by : Angle, optional
            Angle at which to wrap, by default Angle(-360, u.deg)

        Returns
        -------
        Angle
            The Longitude.
        ndarray
            The order for re-ordering other coordinates.
        """
        lt = np.where(lon < cut_at)[0]
        gt = np.where(lon > cut_at)[0]

        order = np.concatenate((gt, lt))
        lon = np.concatenate((lon[gt] + wrap_by, lon[lt]))

        return lon, order

    # ===============================================================

    def in_icrs_frame(
        self,
        *,
        c: CLike = "tab:blue",
        ax: Optional[Axes] = None,
        format_ax: bool = True,
        **kwargs: Any,
    ) -> Axes:
        """Plot stream in an |ICRS| frame.

        Parameters
        ----------
        c : str or array-like[float], optional
            The color or sequence thereof, by default "tab:blue"
        ax : Optional[|Axes|], optional
            Matplotlib |Axes|, by default None
        format_ax : bool, optional
            Whether to add the axes labels and info, by default True

        Returns
        -------
        |Axes|
        """
        stream, _ax, *_ = self._setup(ax)
        kw = self._get_kw(kwargs, label=stream.full_name)

        icrs = stream.coords_ord.frame.transform_to(ICRS())
        icrs.representation_type = SphericalRepresentation

        _ax.scatter(icrs.ra.wrap_at(Angle(180, u.deg)), icrs.dec, c=c, **kw)  # type: ignore

        if format_ax:  # Axes settings
            _ax.set_xlabel(f"RA (ICRS) [{_ax.get_xlabel()}]", fontsize=13)
            _ax.set_ylabel(f"Dec (ICRS) [{_ax.get_ylabel()}]", fontsize=13)
            _ax.grid(True)
            _ax.legend()

        return _ax

    def in_stream_frame(
        self,
        *,
        c: CLike = "tab:blue",
        ax: Optional[Axes] = None,
        format_ax: bool = False,
        **kwargs: Any,
    ) -> Axes:
        """Plot stream in a stream frame.

        Parameters
        ----------
        c : str or array-like[float], optional
            The color or sequence thereof, by default "tab:blue"
        plot_origin : bool, optional
            Whether to plot the origin, by default True
        ax : Optional[|Axes|], optional
            Matplotlib |Axes|, by default None
        format_ax : bool, optional
            Whether to add the axes labels and info, by default True

        Returns
        -------
        |Axes|
        """
        stream, _ax, *_ = self._setup(ax)
        kw = self._get_kw(kwargs, label=stream.full_name)

        sc = stream.coords_ord.transform_to(stream.frame)
        sc.representation_type = SphericalRepresentation

        _ax.scatter(sc.lon, sc.lat, c=c, **kw)  # type: ignore

        if format_ax:  # Axes settings
            _ax.set_xlabel(f"Lon (Stream) [{_ax.get_xlabel()}]", fontsize=13)
            _ax.set_ylabel(f"Lat (Stream) [{_ax.get_ylabel()}]", fontsize=13)
            _ax.grid(True)
            _ax.legend()

        return _ax

    # ========================================================================

    def origin_label_lonlat(self, origin: CoordinateType, *, ax: Optional[Axes]) -> Axes:
        """Label the origin on the plot.

        Parameters
        ----------
        lon : `astropy.coordinates.Longitude`
            The longitude of the origin.
        lat : Quantity
            The latitude of the origin.
        ax : Optional[Axes]
            Matplotlib |Axes|, by default None

        Returns
        -------
        |Axes|
        """
        _, _ax, *_ = self._setup(ax)

        r = origin.represent_as(SphericalRepresentation)
        x = cast(Longitude, r.lon.wrap_at(Angle(180, u.deg))).to_value(u.deg)
        y = r.lat.to_value(u.deg)

        # Plot the central point
        _ax.scatter(x, y, s=10, color="red", label="origin")
        # Add surrounding circle
        _ax.scatter(x, y, s=800, facecolor="None", edgecolor="red")

        return _ax
