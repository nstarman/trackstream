"""Interoperability for `trackstream.track.width.plural.Widths`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar
import warnings

import numpy as np

from trackstream.track.width.plural import WS_FUNCS, Widths

__all__: list[str] = []

if TYPE_CHECKING:
    from collections.abc import Sequence


WS = TypeVar("WS", bound=Widths)


##############################################################################
# CODE
##############################################################################


@WS_FUNCS.implements(np.convolve, dispatch_on=Widths, types=(Widths, np.ndarray))
def convolve(a: WS, v: np.ndarray, mode: Literal["valid", "full", "same"] = "full") -> WS:
    """Return the discrete, linear convolution of widths."""
    # Apply convolution to each contained field.
    ws = {}
    for k, w in a.items():
        ws[k] = np.convolve(w, v, mode=mode)

    return type(a)(ws)


@WS_FUNCS.implements(np.concatenate, dispatch_on=Widths)
def concatenate(
    seqws: Sequence[WS],
    axis: int = 0,
    _: WS | None = None,
    dtype: np.dtype | None = None,
    casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] = "same_kind",
) -> WS:
    """Join a sequence of widths along an existing axis."""
    # Insure
    cls = type(seqws[0])
    if not all(type(ws) is cls for ws in seqws):
        msg = f"widths must all be the same type: {cls}"
        raise ValueError(msg)

    # Get intersecting (& not) keys
    same_keys = set(seqws[0].keys())
    diff_keys = set()
    for ws in seqws:
        diff_keys = same_keys ^ ws.keys()  # symmetric difference
        same_keys &= ws.keys()  # keep only the same

    if any(diff_keys):
        msg = f"non-intersecting keys ({diff_keys}) are not concatenated."
        warnings.warn(msg, stacklevel=2)

    cws = {}
    for k in same_keys:
        cws[k] = np.concatenate(tuple(ws[k] for ws in seqws), axis=axis, out=None, dtype=dtype, casting=casting)

    cls = type(seqws[0])
    return cls(cws)
