# -*- coding: utf-8 -*-

"""Utilities for :mod:`~trackstream.utils`."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from typing import Any, Callable, Dict, Literal, Optional, Protocol, Sequence, Tuple, TypeVar
from typing import Union, runtime_checkable

# THIRD PARTY
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.coordinates import Angle, BaseCoordinateFrame, SkyCoord, SphericalDifferential
from astropy.coordinates import SphericalRepresentation
from astropy.units import Quantity
from astropy.utils.decorators import format_doc
from astropy.utils.misc import indent
from astropy.visualization import quantity_support
from matplotlib.pyplot import Axes

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


def _docstring_indent(s: str) -> Callable[[int], str]:
    def indent_skip(n: int) -> str:
        out: str = indent(s, shift=n)[4 * n :]
        return out

    return indent_skip


# docstrings
_DS: Dict[str, str] = dict(
    DKindT=r"{'positions', 'kinematics'}",
    dkind_ds="The kind of plot.",
)
_DSf: Dict[str, Callable[[int], str]] = dict(
    frame=_docstring_indent(
        """\
A frame instance or its name (a `str`, the default).
Also supported is "stream", which is the stream frame
of the enclosing instance."""
    )
)


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
    def coords_ord(self) -> SkyCoord:
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

    @format_doc(None, frame=_DSf["frame"](3))
    def _parse_frame(self, frame: FrameLikeType, /) -> Tuple[BaseCoordinateFrame, str]:
        """Return the frame and its name.

        Parameters
        ----------
        frame : |Frame| or str, positional-only
            {frame}

        Returns
        -------
        frame : |Frame|
            The parsed frame.
        frame_name : str
            The name of the frame.
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
            The transformed coordinates.
        str
            The name of the frame to which 'crds' have been transformed.
        """
        if frame is None:
            theframe = resolve_framelike(crds)
            name = theframe.__class__.__name__
        else:
            theframe, name = self._parse_frame(frame)

        c = crds.transform_to(theframe)

        if name == "Stream":
            c.representation_type = SphericalRepresentation
            c.differential_type = SphericalDifferential
        elif name == "ICRS":
            c.representation_type = SphericalRepresentation
            c.differential_type = SphericalDifferential

        return c, name

    @format_doc(None, DKindT=_DS["DKindT"], dkind_ds=_DS["dkind_ds"])
    def _get_xy_names(self, frame: CoordinateType, kind: DKindT) -> Tuple[str, str]:
        """Get names of 2D plotting coordinates.

        Parameters
        ----------
        frame : |Frame| or |SkyCoord|
            The frame from which to get the coordinates.
        kind : {DKindT}
            {dkind_ds}

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
            raise ValueError

        # todo reps with 1 dim, like RadialRepresentation
        xn, yn, *_ = tuple(frame.get_representation_component_names(which).keys())

        return xn, yn

    @format_doc(None, DKindT=_DS["DKindT"], dkindt_ds=_DS["dkind_ds"])
    def _get_xy(
        self, crds: CoordinateType, /, kind: DKindT
    ) -> Tuple[Tuple[Quantity, str], Tuple[Quantity, str]]:
        """Get 2D plotting coordinates and names.

        Parameters
        ----------
        crds : CoordinateType
            The coordinates from which to get the 2 dimensions and names.
        kind : {DKindT}
            {dkind_ds}

        Returns
        -------
        Quantity, str
            First set of coordinate and name.
        Quantity, str
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
        ax : Axes
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

        sc, frame_name = self._to_frame(stream.coords_ord, frame=frame)
        (x, xn), (y, yn) = self._get_xy(sc, kind)

        _ax.scatter(x, y, c=c, **kw)

        if format_ax:  # Axes settings
            self._format_ax(_ax, frame=frame_name, x=xn, y=yn)

        return _ax

    @format_doc(None, frame=_DSf["frame"](3), DKindT=_DS["DKindT"], dkind_ds=_DS["dkind_ds"])
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
        origin : |Frame| or |SkyCoord|
            The data to plot.
        frame : |Frame| or str or None, optional
            {frame}
        kind : {DKindT}, optional
            {dkind_ds}
        ax : Optional[Axes]
            Matplotlib |Axes|, by default None

        Returns
        -------
        |Axes|
        """
        _, _ax, *_ = self._setup(ax)

        c, _ = self._to_frame(origin, frame=frame)
        (x, _), (y, _) = self._get_xy(c, kind=kind)

        # Plot the central point
        _ax.scatter(x, y, s=10, color="red", label="origin")
        # Add surrounding circle
        _ax.scatter(x, y, s=800, facecolor="None", edgecolor="red")

        if format_ax:
            _ax.legend()

        return _ax
