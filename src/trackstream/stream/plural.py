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
