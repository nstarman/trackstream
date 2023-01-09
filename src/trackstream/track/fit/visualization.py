"""Stream track plotting."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from typing import TYPE_CHECKING, Any, cast

# THIRD PARTY
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    # THIRD PARTY
    from matplotlib.figure import Figure
    from numpy import ndarray

    # LOCAL
    from trackstream.fit.track.core import StreamArmTrack
    from trackstream.stream.base import StreamBase


__all__ = ["fit_frame_multipanel", "full_multipanel"]


##############################################################################
# CODE
##############################################################################


def fit_frame_multipanel(
    stream: StreamBase,
    *,
    origin: bool = True,
    axes: ndarray | None = None,
    format_ax: bool = True,
    **kwargs: Any,
) -> tuple[Figure, ndarray]:
    """Plot frame fit in a 3 panel plot.

    Parameters
    ----------
    origin : bool, optional keyword-only
        Whether to plot the origin, by default `True`.

    axes : ndarray[|Axes|] or None, optional
        Matplotlib |Axes|. `None` (default) makes a new |Figure| and |Axes|.
    format_ax : bool, optional keyword-only
        Whether to add the axes labels and info, by default `True`.

    **kwargs : Any
        Passed to ``in_frame`` for both positions and kinematics (if
        present).

    Returns
    -------
    |Figure|
        The matplotlib figure.
    ndarray[|Axes|]
        The matplotlib figure axes.

    Raises
    ------
    Exception
        If the stream does not have a fit frame.
    """
    return stream.cache["frame_fit_result"].plot.multipanel(
        stream=stream, origin=origin, axes=axes, format_ax=format_ax, **kwargs
    )


def full_multipanel(
    track: StreamArmTrack,
    *,
    origin: bool = True,
    from_frame_kw: dict[str, Any] | None = None,
    in_frame_kw: dict[str, Any] | None = None,
    som_kw: dict[str, Any] | None = None,
    kalman_kw: dict[str, Any] | None = None,
    format_ax: bool = True,
) -> tuple[Figure, ndarray]:
    """Plot everything.

    Parameters
    ----------
    origin : bool, optional keyword-only
        Whether to plot the origin, by default `True`.

    in_frame_kw : dict[str, Any] or None, optional keyword-only
        Options passed to ``.in_frame()``.
    som_kw : dict[str, Any] or None, optional keyword-only
        Options passed to ``.som()``.
    kalman_kw : dict[str, Any] or None, optional keyword-only
        Options passed to ``.kalman()``.

    format_ax : bool, optional keyword-only
        Whether to add the axes labels and info, by default `True`.

    Returns
    -------
    |Figure|
        The matplotlib figure.
    (5, 2) ndarray[|Axes|]
        The matplotlib figure axes.
    """
    stream = track.stream

    ncols = 2 if track.has_kinematics else 1
    figwidth = 16 if track.has_kinematics else 8
    fig, axs = plt.subplots(5, ncols, figsize=(figwidth, 12))
    axs = cast("ndarray", axs)
    if len(axs.shape) == 1:
        axs.shape = (-1, 1)

    # Plot frame fit
    fit_frame_multipanel(stream, axes=axs[:3, :], format_ax=format_ax, origin=origin, **(from_frame_kw or {}))

    axs[2, 0].clear()
    axs[2, 1].clear()

    track.plot.full_multipanel(
        origin=origin,
        in_frame_kw=in_frame_kw,
        som_kw=som_kw,
        kalman_kw=kalman_kw,
        axes=axs[2:, :],
        format_ax=format_ax,
    )

    return fig, axs
