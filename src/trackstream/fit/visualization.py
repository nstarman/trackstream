# -*- coding: utf-8 -*-

"""Stream track plotting."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, cast

# THIRD PARTY
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.units import Quantity
from matplotlib.figure import Figure
from numpy import ndarray

if TYPE_CHECKING:
    # LOCAL
    from trackstream.fit.track.core import StreamArmTrack
    from trackstream.stream.base import StreamBase


__all__ = ["fit_frame_multipanel", "full_multipanel"]


##############################################################################
# CODE
##############################################################################


def fit_frame_multipanel(
    stream: StreamBase,
    origin: bool = True,
    axes: Optional[ndarray] = None,
    format_ax: bool = True,
    **kwargs: Any,
) -> Tuple[Figure, ndarray]:
    """Plot frame fit in a 3 pandel plot.

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
    return stream.cache["frame_fit"].plot.multipanel(
        stream=stream, origin=origin, axes=axes, format_ax=format_ax, **kwargs
    )


def full_multipanel(
    track: StreamArmTrack,
    *,
    origin: bool = True,
    som_initial_prototypes: bool = False,
    som_prototypes_offset: Quantity = Quantity(0.0, u.deg),
    kalman_kw: Optional[Dict[str, Any]] = None,
    format_ax: bool = True,
) -> Tuple[Figure, ndarray]:
    """Plot everything.

    Parameters
    ----------
    origin : bool, optional keyword-only
        Whether to plot the origin, by default `True`.
    som_initial_prototypes : bool, optional
        Whether to plot the original prototypes, by default False
    som_prototypes_offset : Quantity['angle'], optional keyword-only
        |Latitude| offset for the SOM prototypes.
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
    axs = cast(ndarray, axs)
    if len(axs.shape) == 1:
        axs.shape = (-1, 1)

    # Plot frame fit
    fit_frame_multipanel(stream, axes=axs[:3, :], format_ax=format_ax, origin=origin)

    # clear the last axes, it's replotted
    xl = axs[2, 0].get_xlabel()[:-1]
    axs[2, 0].clear()
    axs[2, 0].set_xlabel(xl[-xl[::-1].find("[") :])

    if axs.shape[1] > 1:
        xl = axs[2, -1].get_xlabel()[:-1]
        axs[2, -1].clear()
        axs[2, -1].set_xlabel(xl[-xl[::-1].find("[") :])

    track.plot.full_multipanel(
        origin=origin,
        som_initial_prototypes=som_initial_prototypes,
        som_prototypes_offset=som_prototypes_offset,
        kalman_kw=kalman_kw,
        axes=axs[2:, :],
        format_ax=format_ax,
    )

    return fig, axs
