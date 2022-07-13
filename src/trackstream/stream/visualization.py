"""Core Functions."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from typing import TYPE_CHECKING, Any, Mapping, TypeVar

# THIRD PARTY
from astropy.visualization import quantity_support
from attrs import define
from matplotlib.pyplot import Axes

# LOCAL
from trackstream._typing import CoordinateType, FrameLikeType
from trackstream.visualization import (
    DKindT,
    PlotCollectionBase,
    StreamPlotDescriptorBase,
)

if TYPE_CHECKING:
    # LOCAL
    from .base import StreamBase
    from .plural import Stream  # noqa: F401


__all__: list[str] = []

##############################################################################
# PARAMETERS

# Ensure Quantity is supported in plots
quantity_support()

# typing
StreamBaseT = TypeVar("StreamBaseT", bound="StreamBase")

##############################################################################
# CODE
##############################################################################


@define(frozen=True)
class StreamBasePlotDescriptor(StreamPlotDescriptorBase[StreamBaseT]):
    """Plot methods for `trackstream.stream.base.StreamBase` objects."""

    def in_frame(
        self,
        frame: str = "ICRS",
        kind: DKindT = "positions",
        *,
        origin: bool = False,
        ax: Axes | None = None,
        format_ax: bool = False,
        **kwargs: Any,
    ) -> Axes:
        """Plot stream in an |ICRS| frame.

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
            Passed to :func:`matplotlib.pyplot.scatter`.

        Returns
        -------
        |Axes|
        """
        _, _ax, *_ = self._setup(ax=ax)

        super().in_frame(frame=frame, kind=kind, ax=_ax, format_ax=format_ax, **kwargs)

        if origin:
            self.origin(frame=frame, kind=kind, ax=_ax, format_ax=format_ax)

        return _ax


#####################################################################


# todo? inherit from StreamBasePlotDescriptor
@define(frozen=True, init=False)
class StreamPlotDescriptor(PlotCollectionBase["Stream"]):
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
        arm0 = next(iter(self._enclosing.values()))
        return arm0.plot.origin(frame=frame, kind=kind, ax=ax, format_ax=format_ax)

    def in_frame(
        self,
        frame: str = "ICRS",
        kind: DKindT = "positions",
        *,
        origin: bool = False,
        ax: Axes | None = None,
        format_ax: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        stream = self._enclosing
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
