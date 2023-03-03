"""Type hints.

This project extensively uses :mod:`~typing` hints.
Note that this is not (necessarily) static typing.
"""


from __future__ import annotations

from typing import Protocol, TypeVar

from astropy.coordinates import BaseCoordinateFrame, SkyCoord
from numpy import floating
from numpy.typing import NBitBase, NDArray

__all__: list[str] = []


##############################################################################
# TYPES
##############################################################################

# -------------------------------------
# Python types

EllipsisType = type(Ellipsis)

# -------------------------------------
# NumPy types

N1 = TypeVar("N1", bound=NBitBase)
N2 = TypeVar("N2", bound=NBitBase)

NDFloat = NDArray[floating[N1]]


# -------------------------------------
# Astropy types

CoordinateType = BaseCoordinateFrame | SkyCoord

FrameLikeType = BaseCoordinateFrame | str


class HasFrame(Protocol):
    @property
    def frame(self) -> BaseCoordinateFrame:
        ...


class SupportsFrame(HasFrame, Protocol):
    @property
    def frame(self) -> BaseCoordinateFrame | None:
        ...
