"""Interoperability for interpolated widths."""

from __future__ import annotations

# STDLIB
from collections.abc import Sequence
from typing import Literal, TypeVar

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
def convolve(a: T, v: np.ndarray, mode: Literal["valid", "full", "same"] = "full") -> T:  # noqa: ARG001
    """Returns the discrete, linear convolution of an interpolated width."""
    raise ValueError


@WB_FUNCS.implements(np.concatenate, dispatch_on=InterpolatedWidth, types=InterpolatedWidth)
def concatenate(
    seqwb: Sequence[T],  # noqa: ARG001
    axis: int = 0,  # noqa: ARG001
    out: T | None = None,  # noqa: ARG001
    dtype: np.dtype | None = None,  # noqa: ARG001
    casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] = "same_kind",  # noqa: ARG001
) -> T:
    """Join a sequence of widths along an existing axis.

    Parameters
    ----------
    seqwb : Sequence[T]
        Sequence of widths to concatenate.
    axis : int, optional
        The axis along which the arrays will be joined. Default is 0.
    out : T | None, optional
        If provided, the destination to place the result. The shape must be correct,
        matching that of what concatenate would have returned if no out argument
        were specified. Default is None.
    dtype : np.dtype | None, optional
        If provided, the destination array will have this dtype. Default is None.
    casting : Literal["no", "equiv", "safe", "same_kind", "unsafe"], optional
        Controls what kind of data casting may occur. Default is "same_kind".

    Returns
    -------
    T
        The concatenated width.
    """
    raise ValueError
