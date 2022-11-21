"""Stream arm classes.

Stream arms are descriptors on a `trackstrea.Stream` class.

"""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

# LOCAL
from trackstream.common import CollectionBase
from trackstream.stream.core import StreamArm
from trackstream.utils.visualization import DKindT, PlotCollectionBase

if TYPE_CHECKING:
    # THIRD PARTY
    from matplotlib.axes import Axes

    # LOCAL
    from trackstream._typing import CoordinateType, FrameLikeType

__all__ = ["StreamArmsBase", "StreamArms"]


##############################################################################


@dataclass
class StreamPlotDescriptor(PlotCollectionBase["StreamArmsBase"]):
    """Stream plot descriptor."""

    # todo move to StreamPlotCollection (DNE)
    def origin(
        self,
        origin: CoordinateType,
        /,
        frame: FrameLikeType | None = None,
        kind: DKindT = "positions",
        *,
        ax: Axes | None,
        format_ax: bool = True,
    ) -> Axes:
        """Plot the origin of the stream.

        Parameters
        ----------
        origin : CoordinateType
            The origin of the stream.
        frame : FrameLikeType, optional
            The frame of the origin. If not provided, the frame of the stream
            will be used.
        kind : DKindT, optional
            The kind of data to plot. This can be ``"positions"``,
            ``"velocities"``
        ax : Axes, optional
            The axes to plot on. If not provided, the current axes will be used.
        format_ax : bool, optional
            Whether to format the axes.

        Returns
        -------
        Axes
            The axes that was plotted on.
        """
        # FIXME! plot all origins, but detect and skip repeats
        arm0 = next(iter(self.enclosing.values()))
        return arm0.plot.origin(frame=frame, kind=kind, ax=ax, format_ax=format_ax)

    def in_frame(
        self,
        frame: str = "icrs",
        kind: DKindT = "positions",
        *,
        origin: bool = False,
        ax: Axes | None = None,
        format_ax: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Plot the stream in a given frame.

        Parameters
        ----------
        frame : str
            The frame to plot in.
        kind : DKindT
            The kind of data to plot.
        origin : bool
            Whether to plot the origin of the stream.
        ax : Axes | None
            The axes to plot on.
        format_ax : bool
            Whether to format the axes.
        kwargs : Any
            Keyword arguments to pass to `~trackstream.Stream.plot`.

        Returns
        -------
        dict[str, Any]
            Dictionary  of return values of `~trackstream.Stream.plot`.
        """
        stream = self.enclosing
        last = len(stream.arms) - 1

        out = {}
        for i, n in enumerate(stream.keys()):
            # allow for arm-specific kwargs.
            kw = {k: (v[n] if (isinstance(v, Mapping) and n in v) else v) for k, v in kwargs.items()}

            out[n] = stream[n].plot.in_frame(
                frame=frame,
                kind=kind,
                origin=False if i != last else origin,
                ax=ax,
                format_ax=False if i != last else format_ax,
                **kw,
            )

        return out


##############################################################################


class StreamArmsBase(CollectionBase[StreamArm]):
    """Base class for a collection of stream arms."""


class StreamArms(StreamArmsBase):
    """A collection of stream arms.

    See Also
    --------
    Stream
        An object that brings together 2 stream arms, but acts like 1 stream.
    """
