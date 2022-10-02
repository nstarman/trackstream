##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from typing import Literal, Sequence, TypeVar

# THIRD PARTY
import numpy as np

# LOCAL
from trackstream.track.width.base import WB_FUNCS
from trackstream.track.width.interpolated import InterpolatedWidth

__all__: list[str] = []


##############################################################################
# TYPING

T = TypeVar("T")


##############################################################################
# CODE
##############################################################################


@WB_FUNCS.implements(np.convolve, dispatch_on=InterpolatedWidth, types=(InterpolatedWidth, np.ndarray))
def convolve(a: T, v: np.ndarray, mode: Literal["valid", "full", "same"] = "full") -> T:
    raise ValueError


@WB_FUNCS.implements(np.concatenate, dispatch_on=InterpolatedWidth, types=InterpolatedWidth)
def concatenate(
    seqwb: Sequence[T],
    axis: int = 0,
    out: T | None = None,
    dtype: np.dtype | None = None,
    casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] = "same_kind",
) -> T:
    """Join a sequence of arrays along an existing axis."""
    raise ValueError
