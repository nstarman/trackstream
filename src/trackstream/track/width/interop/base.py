"""Interoperability."""

from __future__ import annotations

# STDLIB
from collections.abc import Sequence
from dataclasses import fields, replace
from typing import Literal, TypeVar

# THIRD PARTY
import numpy as np

# LOCAL
from trackstream.track.width.base import WB_FUNCS, WidthBase

__all__: list[str] = []


##############################################################################
# TYPING

T = TypeVar("T")


##############################################################################
# CODE
##############################################################################


@WB_FUNCS.implements(np.convolve, WidthBase, types=(WidthBase, np.ndarray))
def convolve(a: T, v: np.ndarray, mode: Literal["valid", "full", "same"] = "full") -> T:
    """Returns the discrete, linear convolution of two one-dimensional sequences."""
    # apply to each field.
    return replace(a, **{f.name: np.convolve(getattr(a, f.name), v, mode=mode) for f in fields(a)})


@WB_FUNCS.implements(np.concatenate, WidthBase, types=WidthBase)
def concatenate(
    seqwb: Sequence[T],
    axis: int = 0,
    out: T | None = None,
    dtype: np.dtype | None = None,
    casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] = "same_kind",
) -> T:
    """Join a sequence of arrays along an existing axis."""
    # Check types are the same
    cls = type(seqwb[0])
    if not all(type(wb) is cls for wb in seqwb):
        raise ValueError(f"widths must all be the same type: {cls}")

    N = len(seqwb)
    if N == 0:
        raise ValueError("need at least one array to concatenate")

    if out is not None:
        raise ValueError("out must be None")
    elif dtype is not None:
        raise ValueError("dtype must be None")

    if N == 1:
        if axis != 0:
            raise ValueError("axis must be 0 for 1 width")
        return seqwb[0]
    # else:  N == 2

    # build concatenation of each field
    fs = {f.name: np.concatenate(tuple(getattr(wb, f.name) for wb in seqwb)) for f in fields(cls)}
    return cls(**fs)
