"""Utilities for :mod:`~trackstream.utils`."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from dataclasses import dataclass
from typing import Any

# THIRD PARTY
from astropy.coordinates import BaseCoordinateFrame
from matplotlib.pyplot import Axes

# LOCAL
from trackstream._typing import CoordinateType, FrameLikeType
from trackstream.utils.coord_utils import parse_framelike
from trackstream.utils.descriptors.base import BndTo
from trackstream.utils.visualization import AX_LABELS, CommonPlotDescriptorBase, DKindT

__all__: list[str] = []


##############################################################################
# CODE
##############################################################################


@dataclass(frozen=True)
class StreamPlotDescriptorBase(CommonPlotDescriptorBase[BndTo]):
    """ABC descriptor for plotting stream(arms).

    Parameters
    ----------
    default_scatter_kwargs: dict[str, Any] or None, optional keyword-only
        Default keyword arguments for :func:`matplotlib.pyplot.scatter`.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.default_scatter_kwargs.setdefault("marker", "*")

    def _parse_frame(self, frame: FrameLikeType, /) -> tuple[BaseCoordinateFrame, str]:
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
            theframe = frame
            frame_name = frame.__class__.__name__
        # must be a str
        elif frame.lower() == "stream":
            frame_name = "Stream"
            maybeframe = self.enclosing.frame
            if maybeframe is None:
                # LOCAL
                from .base import FRAME_NONE_ERR

                raise FRAME_NONE_ERR
            theframe = maybeframe
        else:
            theframe = parse_framelike(frame)
            frame_name = frame.__class__.__name__

        return theframe, frame_name

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
        c, name = super()._to_frame(crds, frame=frame)

        if name == "Stream":
            frame = self.enclosing.frame

            if frame is None:
                # LOCAL
                from .base import FRAME_NONE_ERR

                raise FRAME_NONE_ERR

            # set frame, rep
            c.representation_type = frame.representation_type
            c.differential_type = frame.differential_type

        return c, name

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
        frame: str = "icrs",
        kind: DKindT = "positions",
        *,
        ax: Axes | None = None,
        format_ax: bool = False,
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

        sc, frame_name = self._to_frame(stream.coords, frame=frame)
        (x, xn), (y, yn) = self._get_xy(sc, kind)

        _ax.scatter(x, y, **kw)

        if format_ax:  # Axes settings
            self._format_ax(_ax, frame=frame_name, x=xn, y=yn)

        return _ax

    def origin(
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
