# -*- coding: utf-8 -*-

"""Utilities for :mod:`~trackstream.utils`."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from typing import (
    Any,
    Dict,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    get_args,
    overload,
    runtime_checkable,
)

# THIRD PARTY
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.coordinates import (
    Angle,
    BaseCoordinateFrame,
    SkyCoord,
    SphericalDifferential,
    SphericalRepresentation,
)
from astropy.units import Quantity
from astropy.visualization import quantity_support
from matplotlib.pyplot import Axes
from typing_extensions import Unpack

# LOCAL
from trackstream._type_hints import CoordinateType, FrameLikeType
from trackstream.utils.coord_utils import resolve_framelike
from trackstream.utils.descriptors import EnclType, InstanceDescriptor

##############################################################################
# PARAMETERS

# Ensure Quantity is supported in plots
quantity_support()

# Types
CLike = Union[str, Sequence[float], Quantity]
DKindT = Union[Literal["positions"], Literal["kinematics"]]
StreamLikeType = TypeVar("StreamLikeType", bound="StreamLike")


# Axes labels
AX_LABELS = {
    "ra": "RA",
    "dec": "Dec",
    "pm_ra": r"$\mu_{\alpha}$",
    "pm_ra_cosdec": r"$\mu_{\alpha}^{*}$",
    "pm_dec": r"$\mu_{\delta}$",
    "lon": r"$\ell$",
    "lat": r"$b$",
    "pm_lon": r"$\mu_{\ell}$",
    "pm_lon_coslat": r"$\mu_{\ell}^{*}$",
    "pm_lat": r"$\mu_{b}$",
}


# # docstrings
# _DS: Dict[str, str] = dict()
# _DSf: Dict[str, Callable[[int], str]] = dict()


##############################################################################
# CODE
##############################################################################


class PlotDescriptorBase(InstanceDescriptor[EnclType]):
    """Plot descriptor base class.

    Parameters
    ----------
    default_scatter_kwargs: dict[str, Any] or None, optional keyword-only
        Default keyword arguments for :func:`matplotlib.pyplot.scatter`.
    """

    _default_scatter_kwargs: Dict[str, Any]

    def __init__(self, *, default_scatter_kwargs: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()

        # Default scatter style
        scatter_style = default_scatter_kwargs or {}
        scatter_style.setdefault("s", 3)
        self._default_scatter_kwargs = scatter_style

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
            Mix of ``kwargs``, ``defaults``, and ``_default_scatter_kwargs``,
            preferring them in that order.
        """
        kw: Dict[str, Any] = {**self._default_scatter_kwargs, **defaults, **(kwargs or {})}
        return kw

    @overload
    def _setup(self, *, ax: Axes) -> Tuple[EnclType, Axes, Unpack[Tuple[Any, ...]]]:
        ...

    @overload
    def _setup(self, *, ax: None) -> Tuple[EnclType, Axes, Unpack[Tuple[Any, ...]]]:
        ...

    @overload
    def _setup(self, *, ax: Literal[False]) -> Tuple[EnclType, None, Unpack[Tuple[Any, ...]]]:
        ...

    def _setup(
        self, *, ax: Union[Axes, None, Literal[False]] = None
    ) -> Tuple[EnclType, Optional[Axes], Unpack[Tuple[Any, ...]]]:
        """Setup the plot.

        Parameters
        ----------
        ax : |Axes| or None, optional keyword-only
            Matplotlib |Axes|. `None` (default) uses the current axes
            (:func:`matplotlib.pyplot.gca`).

        Returns
        -------
        tuple[Any, ...]
            At least (`trackstream.visualization.StreamLike`, |Axes|)
        """
        # Stream
        parent = self._enclosing
        # Get matplotlib axes
        if ax is False:
            _ax = None
        elif ax is None:
            _ax = plt.gca()
        else:
            _ax = ax

        return parent, _ax


##############################################################################


@runtime_checkable
class StreamLike(Protocol):
    """Stream-like Protocol."""

    @property
    def full_name(self) -> Optional[str]:
        ...

    @property
    def coords_ord(self) -> SkyCoord:
        ...

    @property
    def frame(self) -> BaseCoordinateFrame:
        ...


class StreamPlotDescriptorBase(PlotDescriptorBase[StreamLikeType]):
    """Plot descriptor base class.

    Parameters
    ----------
    default_scatter_kwargs: dict[str, Any] or None, optional keyword-only
        Default keyword arguments for :func:`matplotlib.pyplot.scatter`.
    """

    def __init__(self, *, default_scatter_kwargs: Optional[Dict[str, Any]] = None) -> None:
        # Default scatter style
        scatter_style = default_scatter_kwargs or {}
        scatter_style.setdefault("marker", "*")

        super().__init__(default_scatter_kwargs=scatter_style)

    def _parse_frame(self, frame: FrameLikeType, /) -> Tuple[BaseCoordinateFrame, str]:
        """Return the frame and its name.

        Parameters
        ----------
        frame : |Frame| or str, positional-only
            A frame instance or its name (a `str`, the default).
            Also supported is "stream", which is the stream frame
            of the enclosing instance.

        Returns
        -------
        frame : |Frame|
            The parsed frame.
        frame_name : str
            The name of the parsed frame.
        """
        if not isinstance(frame, (BaseCoordinateFrame, str)):
            raise ValueError(f"{frame} is not a BaseCoordinateFrame or str")

        if isinstance(frame, BaseCoordinateFrame):
            frame_name = frame.__class__.__name__
        # must be a str
        elif frame.lower() == "stream":
            frame_name = "Stream"
            frame = self._enclosing.frame
        else:
            frame = resolve_framelike(frame)
            frame_name = frame.__class__.__name__

        return frame, frame_name

    def _to_frame(
        self, crds: CoordinateType, frame: Optional[FrameLikeType] = None
    ) -> Tuple[CoordinateType, str]:
        """Transform coordinates to a frame.

        Parameters
        ----------
        crds : BaseCoordinateFrame or SkyCoord
            The coordinates to transform.
        frame : |Frame| or str or None
            The frame to which to transform `crds`. If `None`, `crds` are not
            tranformed.

        Returns
        -------
        |Frame| or |SkyCoord|
            The transformed coordinates. Output type matches input type.
        str
            The name of the frame to which 'crds' have been transformed.
        """
        if frame is None:
            theframe = resolve_framelike(crds)
            name = theframe.__class__.__name__
        else:
            theframe, name = self._parse_frame(frame)

        # TODO! report bug in SkyCoord requiring `.frame`
        c = SkyCoord(crds, copy=False).frame.transform_to(theframe)

        if name == "Stream":
            c.representation_type = SphericalRepresentation
            c.differential_type = SphericalDifferential
        elif name == "ICRS":
            c.representation_type = SphericalRepresentation
            c.differential_type = SphericalDifferential

        return c, name

    def _get_xy_names(self, frame: CoordinateType, kind: DKindT) -> Tuple[str, str]:
        """Get names of 2D plotting coordinates.

        Parameters
        ----------
        frame : |Frame| or |SkyCoord|
            The frame from which to get the coordinates.
        kind : {'positions', 'kinematics'}
            The kind of plot.

        Returns
        -------
        str, str
            Names of the 2D plotting coordinates.
        """
        if kind == "positions":
            which = "base"
        elif kind == "kinematics":
            which = "s"
        else:
            raise ValueError(f"kind must be in {get_args(DKindT)}, not {kind!r}")

        # todo reps with 1 dim, like RadialRepresentation
        xn, yn, *_ = tuple(frame.get_representation_component_names(which).keys())

        return xn, yn

    def _get_xy(
        self, crds: CoordinateType, /, kind: DKindT
    ) -> Tuple[Tuple[Quantity, str], Tuple[Quantity, str]]:
        """Get 2D plotting coordinates and names.

        Parameters
        ----------
        crds : CoordinateType
            The coordinates from which to get the 2 dimensions and names.
        kind : {'positions', 'kinematics'}
            The kind of plot.

        Returns
        -------
        |Quantity|, str
            First set of coordinate and name.
        |Quantity|, str
            Second set of coordinate and name.
        """
        xn, yn = self._get_xy_names(crds, kind=kind)

        x = getattr(crds, xn)
        if isinstance(x, Angle):
            x = x.wrap_at(Angle(180, u.deg))

        y = getattr(crds, yn)

        return (x, xn), (y, yn)

    def _format_ax(self, ax: Axes, /, *, frame: str, x: str, y: str) -> None:
        """Format axes, setting labels and legend.

        Parameters
        ----------
        ax : |Axes| or None, positional-only
            Matplotlib |Axes|. `None` (default) uses the current axes
            (:func:`matplotlib.pyplot.gca`).
        frame : str
            The name of the |Frame|.
        x, y : str
            x and y axis labels, respectively.
        """
        ax.set_xlabel(f"{AX_LABELS.get(x, x)} ({frame}) [{ax.get_xlabel()}]", fontsize=13)
        ax.set_ylabel(f"{AX_LABELS.get(y, y)} ({frame}) [{ax.get_ylabel()}]", fontsize=13)
        ax.grid(True)
        ax.legend()

    # def _wrap_lon_order(
    #     self,
    #     lon: Angle,
    #     cut_at: Angle = Angle(100, u.deg),
    #     wrap_by: Angle = Angle(-360, u.deg),
    # ) -> Tuple[Angle, np.ndarray]:
    #     """Wrap the stream by `~astropy.coordinates.Longitude`.

    #     Parameters
    #     ----------
    #     lon : Angle
    #         Longitude.
    #     cut_at : Angle, optional
    #         Angle at which to cut, by default Angle(100, u.deg)
    #     wrap_by : Angle, optional
    #         Angle at which to wrap, by default Angle(-360, u.deg)

    #     Returns
    #     -------
    #     Angle
    #         The Longitude.
    #     ndarray
    #         The order for re-ordering other coordinates.
    #     """
    #     lt = np.where(lon < cut_at)[0]
    #     gt = np.where(lon > cut_at)[0]

    #     order = np.concatenate((gt, lt))
    #     lon = np.concatenate((lon[gt] + wrap_by, lon[lt]))

    #     return lon, order

    # ===============================================================

    def in_frame(
        self,
        frame: str = "ICRS",
        kind: DKindT = "positions",
        *,
        ax: Optional[Axes] = None,
        format_ax: bool = True,
        **kwargs: Any,
    ) -> Axes:
        """Plot stream in an |ICRS| frame.

        Parameters
        ----------
        frame : |Frame| or |SkyCoord|, optional
            The frame from which to get the coordinates.
        kind : {'positions', 'kinematics'}, optional
            The kind of plot.

        ax : |Axes| or None, optional keyword-only
            Matplotlib |Axes|. `None` (default) uses the current axes
            (:func:`matplotlib.pyplot.gca`).
        format_ax : bool, optional keyword-only
            Whether to add the axes labels and info, by default `True`.
        **kwargs : Any
            Passed to :func:`matplotlib.pyplot.scatter`.

        Returns
        -------
        |Axes|
        """
        stream, _ax, *_ = self._setup(ax=ax)
        kw = self._get_kw(kwargs, label=stream.full_name)

        sc, frame_name = self._to_frame(stream.coords_ord, frame=frame)
        (x, xn), (y, yn) = self._get_xy(sc, kind)

        _ax.scatter(x, y, **kw)

        if format_ax:  # Axes settings
            self._format_ax(_ax, frame=frame_name, x=xn, y=yn)

        return _ax

    def origin(
        self,
        origin: CoordinateType,
        /,
        frame: Optional[FrameLikeType] = None,
        kind: DKindT = "positions",
        *,
        ax: Optional[Axes],
        format_ax: bool = True,
    ) -> Axes:
        """Label the origin on the plot.

        Parameters
        ----------
        origin : |Frame| or |SkyCoord|, positional-only
            The data to plot.

        frame : |Frame| or str or None, optional
            A frame instance or its name (a `str`, the default).
            Also supported is "stream", which is the stream frame
            of the enclosing instance.
        kind : {'positions', 'kinematics'}, optional
            The kind of plot.

        ax : |Axes| or None, optional keyword-only
            Matplotlib |Axes|. `None` (default) uses the current axes
            (:func:`matplotlib.pyplot.gca`).
        format_ax : bool, optional keyword-only
            Whether to add the axes labels and info, by default `True`.

        Returns
        -------
        |Axes|
        """
        _, _ax, *_ = self._setup(ax=ax)

        c, _ = self._to_frame(origin, frame=frame)
        (x, _), (y, _) = self._get_xy(c, kind=kind)

        # Plot the central point
        _ax.scatter(x, y, s=10, color="red", label="origin")
        # Add surrounding circle
        _ax.scatter(x, y, s=800, facecolor="None", edgecolor="red")

        if format_ax:
            _ax.legend()

        return _ax