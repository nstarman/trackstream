"""Interoperability."""

from __future__ import annotations

from dataclasses import fields, replace
from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np

from trackstream.track.width.base import WB_FUNCS, WidthBase

__all__: list[str] = []

if TYPE_CHECKING:
    from collections.abc import Sequence

    T = TypeVar("T")


##############################################################################
# CODE
##############################################################################


@WB_FUNCS.implements(np.convolve, WidthBase, types=(WidthBase, np.ndarray))
def convolve(a: T, v: np.ndarray, mode: Literal["valid", "full", "same"] = "full") -> T:
    """Return the discrete, linear convolution of two one-dimensional sequences."""
    # apply to each field.
    return replace(a, **{f.name: np.convolve(getattr(a, f.name), v, mode=mode) for f in fields(a)})


@WB_FUNCS.implements(np.concatenate, WidthBase, types=WidthBase)
def concatenate(
    seqwb: Sequence[T],
    axis: int = 0,
    out: T | None = None,
    dtype: np.dtype | None = None,
    casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] = "same_kind",  # noqa: ARG001
) -> T:
    """Join a sequence of arrays along an existing axis."""
    # Check types are the same
    cls = type(seqwb[0])
    if not all(type(wb) is cls for wb in seqwb):
        msg = f"widths must all be the same type: {cls}"
        raise ValueError(msg)

    N = len(seqwb)
    if N == 0:
        msg = "need at least one array to concatenate"
        raise ValueError(msg)

    if out is not None:
        msg = "out must be None"
        raise ValueError
    if dtype is not None:
        msg = "dtype must be None"
        raise ValueError(msg)

    if N == 1:
        if axis != 0:
            msg = "axis must be 0 for 1 width"
            raise ValueError(msg)
        return seqwb[0]

    # build concatenation of each field
    fs = {f.name: np.concatenate(tuple(getattr(wb, f.name) for wb in seqwb)) for f in fields(cls)}
    return cls(**fs)
