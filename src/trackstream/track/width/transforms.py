##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from dataclasses import replace
from functools import singledispatch
from typing import Any, Callable, NoReturn, Protocol, TypeVar

# THIRD PARTY
from astropy.coordinates import BaseRepresentation

# LOCAL
from trackstream.track.width.base import WidthBase
from trackstream.track.width.core import BaseWidth
from trackstream.track.width.interpolated import InterpolatedWidth

__all__: list[str] = []

##############################################################################
# TYPING


W1 = TypeVar("W1", bound="WidthBase")
W2 = TypeVar("W2", bound="WidthBase")
BW = TypeVar("BW", bound="BaseWidth")


##############################################################################
# CODE
##############################################################################


@singledispatch
def represent_as(w1: object, w2type: type[W2], point: BaseRepresentation) -> NoReturn:
    raise NotImplementedError("not dispatched")


@represent_as.register
def _represent_as_plainwidth(w1: BaseWidth, w2type: type[W2], point: BaseRepresentation) -> W2:
    func = WIDTH_TRANSFORMATIONS[(w1.__class__, w2type)]
    attrs = func(w1, point)
    return w2type(**attrs)


@represent_as.register
def _represent_as_interpolatedwidth(
    w1: InterpolatedWidth, w2type: type[BW], point: BaseRepresentation
) -> InterpolatedWidth[BW]:
    width = represent_as(w1.width, w2type, point)
    return replace(w1, width=width)


# ===================================================================


WIDTH_TRANSFORMATIONS: dict[tuple[type, type], Callable[[Any, BaseRepresentation], Any]] = {}  # TODO! not Any


class Transformer(Protocol):
    def __call__(self, cw: WidthBase, point: BaseRepresentation) -> dict[str, Any]:  # TODO! typeddict
        ...


def register_transformation(w1type: type[W1], w2type: type[W2]) -> Callable[[Transformer], Transformer]:
    def decorator(func: Transformer) -> Transformer:
        WIDTH_TRANSFORMATIONS[(w1type, w2type)] = func
        return func

    return decorator
