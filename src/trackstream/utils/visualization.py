"""Utilities for :mod:`~trackstream.utils`."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import inspect
from collections.abc import Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast, get_args, overload

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
from bound_class.core.descriptors import BoundDescriptor

# LOCAL
from trackstream.utils.coord_utils import parse_framelike

if TYPE_CHECKING:
    # THIRD PARTY
    from matplotlib.pyplot import Axes
    from typing_extensions import Unpack

    # LOCAL
    from trackstream._typing import CoordinateType, FrameLikeType
    from trackstream.common import CollectionBase


__all__: list[str] = []


##############################################################################
# PARAMETERS

# Ensure Quantity is supported in plots
quantity_support()

# Types
BndTo = TypeVar("BndTo")
CollectionBaseT = TypeVar("CollectionBaseT", bound="CollectionBase")
CLike = str | Sequence[float] | Quantity
DKindT = Literal["positions"] | Literal["kinematics"]


AX_LABELS = MappingProxyType(
    {
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
        "v_x": r"$v_x$",
        "v_y": r"$v_y$",
        "v_z": r"$v_z$",
    }
)
"""`matplotlib.pyplot.Axes` label substitutions. Read only."""


##############################################################################
# CODE
##############################################################################


@dataclass
class PlotDescriptorBase(BoundDescriptor[BndTo]):
    """Plot descriptor base class.

    Parameters
    ----------
    default_scatter_kwargs: dict[str, Any] or None, optional keyword-only
        Default keyword arguments for :func:`matplotlib.pyplot.scatter`.
    """

    default_scatter_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.default_scatter_kwargs.setdefault("s", 3)

    def _get_kw(self, kwargs: dict[str, Any] | None = None, **defaults: Any) -> dict[str, Any]:
        """Get plot options.

        Parameters
        ----------
        kwargs : dict[str, Any]
            Plot options.
        **defaults: Any
            Default plot options. ``kwargs`` takes precedence.

        Returns
        -------
        dict[str, Any]
            Mix of ``kwargs``, ``defaults``, and ``default_scatter_kwargs``,
            preferring them in that order.
        """
        # c and color overlap
        if "c" in defaults and "color" in kwargs:
            defaults.pop("c")
        elif "color" in defaults and "c" in kwargs:
            defaults.pop("color")

        kw: dict[str, Any] = {**self.default_scatter_kwargs, **defaults, **(kwargs or {})}
        return kw

    @overload
    def _setup(self, *, ax: Axes) -> tuple[BndTo, Axes, Unpack[tuple[Any, ...]]]:
        ...

    @overload
    def _setup(self, *, ax: None) -> tuple[BndTo, Axes, Unpack[tuple[Any, ...]]]:
        ...

    @overload
    def _setup(self, *, ax: Literal[False]) -> tuple[BndTo, None, Unpack[tuple[Any, ...]]]:
        ...

    @overload
    def _setup(self, *, ax: bool) -> tuple[BndTo, Axes, Unpack[tuple[Any, ...]]]:
        ...

    def _setup(self, *, ax: Axes | None | bool = None) -> tuple[BndTo, Axes | None, Unpack[tuple[Any, ...]]]:
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
        parent = self.enclosing
        # Get matplotlib axes
        _ax: Axes | None
        if ax is False:
            _ax = None
        elif ax is True or ax is None:
            _ax = plt.gca()
        else:
            _ax = ax

        return parent, _ax, None

    def _get_xy_names(self, frame: CoordinateType, kind: DKindT) -> tuple[str, str]:
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

        # TODO! reps with 1 dim, like RadialRepresentation
        namedict = cast("dict", frame.get_representation_component_names(which))
        xn, yn, *_ = tuple(namedict.keys())

        return xn, yn

    def _get_xy(self, crds: CoordinateType, /, kind: DKindT) -> tuple[tuple[Quantity, str], tuple[Quantity, str]]:
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
        x = cast("Quantity", x)

        y = cast("Quantity", getattr(crds, yn))

        return (x, xn), (y, yn)


@dataclass
class CommonPlotDescriptorBase(PlotDescriptorBase[BndTo]):
    """Common plot descriptor base class."""

    def _parse_frame(self, framelike: FrameLikeType, /) -> tuple[BaseCoordinateFrame, str]:
        """Return the frame and its name.

        Parameters
        ----------
        framelike : |Frame| or str, positional-only
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
        if not isinstance(framelike, (BaseCoordinateFrame, str)):
            raise ValueError(f"{framelike} is not a BaseCoordinateFrame or str")

        if isinstance(framelike, BaseCoordinateFrame):
            frame = framelike
            frame_name = framelike.__class__.__name__
        # must be a str
        else:
            frame = parse_framelike(framelike)
            frame_name = frame.__class__.__name__

        return frame, frame_name

    def _to_frame(self, crds: CoordinateType, frame: FrameLikeType | None = None) -> tuple[CoordinateType, str]:
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
            theframe = parse_framelike(crds)
            name = theframe.__class__.__name__
        else:
            theframe, name = self._parse_frame(frame)

        # TODO! report bug in SkyCoord requiring `.frame`
        c = SkyCoord(crds, copy=False).frame.transform_to(theframe)

        # known shortcut
        if name == "icrs":
            c.representation_type = SphericalRepresentation
            c.differential_type = SphericalDifferential

        return c, name

    def _origin(
        self,
        frame: FrameLikeType | None = None,
        kind: DKindT = "positions",
        *,
        ax: Axes | None = None,
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
        obj, _ax, *_ = self._setup(ax=ax)

        c, _ = self._to_frame(obj.origin, frame=frame)
        (x, _), (y, _) = self._get_xy(c, kind=kind)

        # Plot the central point
        _ax.scatter(x, y, s=10, color="red", label="origin")
        # Add surrounding circle
        _ax.scatter(x, y, s=800, facecolor="None", edgecolor="red")

        if format_ax:
            _ax.legend()

        return _ax

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
        # ax.grid(True)
        ax.legend()


##############################################################################

# todo: make a subclass of CollectionBase
@dataclass
class PlotCollectionBase(BoundDescriptor[CollectionBaseT]):
    """Base class for plotting collections."""

    def __iter__(self):
        enclosing = self.enclosing
        yield from (enclosing[k].plot for k in enclosing.keys())

    def __getitem__(self, key: str) -> PlotDescriptorBase:
        return self.enclosing[key].plot

    def __getattr__(self, key: str) -> Any:
        if key in ("__isabstractmethod__",):
            return object.__getattribute__(self, key)
        try:
            enclosing = self.enclosing
        except ValueError as e:
            raise AttributeError from e

        k0 = tuple(enclosing.keys())[0]

        # if getting a method, broadcast to each stream.
        if not callable(getattr(self[k0], key)):
            raise NotImplementedError

        def apply(*args: Any, **kwargs: Any) -> dict[str, Any]:
            # check the first
            method = getattr(self[tuple(enclosing.keys())[0]], key)
            sig = inspect.signature(method)
            mainba = sig.bind_partial(*args, **kwargs)

            # for all keyword-only, broadcast to the last plotter.
            for n, v in mainba.arguments.items():
                param = sig.parameters[n]
                if param.kind >= 3 and not isinstance(v, dict):
                    mainba.arguments[n] = {k: v for k in enclosing.keys()}

            out = {}
            for name in enclosing.keys():
                method = getattr(self[name], key)
                sig = inspect.signature(method)
                ba = sig.bind_partial(*mainba.args, **mainba.kwargs)

                for k, v in ba.arguments.items():
                    if isinstance(v, dict) and name in v:
                        ba.arguments[k] = v[name]

                out[name] = method(*ba.args, **ba.kwargs)

            return out

        return apply
