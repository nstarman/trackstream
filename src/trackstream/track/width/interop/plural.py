##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import warnings
from typing import Literal, Sequence, TypeVar

# THIRD PARTY
import numpy as np

# LOCAL
from trackstream.track.width.plural import Widths
from trackstream.utils.numpy_overload import NumPyOverloader

__all__: list[str] = []


##############################################################################
# TYPING

WS = TypeVar("WS", bound=Widths)


##############################################################################
# CODE
##############################################################################

WS_FUNCS = NumPyOverloader(default_dispatch_on=Widths)

# ============================================================================


@WS_FUNCS.implements(np.convolve, types=(Widths, np.ndarray))
def convolve(a: WS, v: np.ndarray, mode: Literal["valid", "full", "same"] = "full") -> WS:
    # Apply convolution to each contained field.
    ws = {}
    for k, w in a.items():
        ws[k] = np.convolve(w, v, mode=mode)

    return type(a)(ws)


@WS_FUNCS.implements(np.concatenate, types=Widths)
def concatenate(
    seqws: Sequence[WS],
    axis: int = 0,
    out: WS | None = None,
    dtype: np.dtype | None = None,
    casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] = "same_kind",
) -> WS:
    # Insure
    cls = type(seqws[0])
    if not all(type(ws) is cls for ws in seqws):
        raise ValueError(f"widths must all be the same type: {cls}")

    # Get intersecting (& not) keys
    same_keys = set(seqws[0].keys())
    diff_keys = set()
    for ws in seqws:
        diff_keys = same_keys ^ ws.keys()  # symmetric difference
        same_keys &= ws.keys()  # keep only the same

    if any(diff_keys):
        warnings.warn(f"non-intersecting keys ({diff_keys}) are not concatenated.")

    cws = {}
    for k in same_keys:
        cws[k] = np.concatenate(tuple(ws[k] for ws in seqws), axis=axis, out=None, dtype=dtype, casting=casting)

    cls = type(seqws[0])
    return cls(cws)
