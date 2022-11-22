"""Width transformations."""

from __future__ import annotations

# STDLIB
from dataclasses import replace
from functools import singledispatch
from typing import TYPE_CHECKING, Any, Callable, Protocol, TypeVar

# LOCAL
from trackstream.track.width.core import BaseWidth
from trackstream.track.width.interpolated import InterpolatedWidth

if TYPE_CHECKING:
    # THIRD PARTY
    from astropy.coordinates import BaseRepresentation

    # LOCAL
    from trackstream.track.width.base import WidthBase

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
def represent_as(
    w1: object, w2type: type[W2], point: BaseRepresentation
) -> Any:  # https://github.com/python/mypy/issues/11727
    """Represent as a new width type."""
    raise NotImplementedError("not dispatched")


@represent_as.register(BaseWidth)
def _represent_as_plainwidth(w1: BaseWidth, w2type: type[W2], point: BaseRepresentation) -> W2:
    func = WIDTH_TRANSFORMATIONS[(w1.__class__, w2type)]
    w2 = func(w1, point)
    return w2


@represent_as.register(InterpolatedWidth)
def _represent_as_interpolatedwidth(
    w1: InterpolatedWidth, w2type: type[BW], point: BaseRepresentation
) -> InterpolatedWidth[BW]:
    width = represent_as(w1.width, w2type, point)
    return replace(w1, width=width)


# ===================================================================


WIDTH_TRANSFORMATIONS: dict[tuple[type, type], Callable[[Any, BaseRepresentation], Any]] = {}


class Transformer(Protocol[W1, W2]):
    """Width transformation callable."""

    def __call__(self, cw: W1, point: BaseRepresentation) -> W2:
        """Transform a width."""
        ...


def register_transformation(w1type: type[W1], w2type: type[W2]) -> Callable[[Transformer[W1, W2]], Transformer[W1, W2]]:
    """Register a width transformation."""

    def decorator(func: Transformer[W1, W2]) -> Transformer[W1, W2]:
        """Decorator."""
        WIDTH_TRANSFORMATIONS[(w1type, w2type)] = func
        return func

    return decorator
