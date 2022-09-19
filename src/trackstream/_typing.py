"""Type hints.

This project extensively uses :mod:`~typing` hints.
Note that this is not (necessarily) static typing.
"""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import os
from typing import Protocol, Union

# THIRD PARTY
import astropy.units as u
import numpy as np
from astropy.coordinates import BaseCoordinateFrame, BaseRepresentation, SkyCoord

__all__ = [
    "EllipsisType",
    "FullPathLike",
    "ArrayLike",
    "CoordinateType",
    "CoordinateLikeType",
    "FrameLikeType",
    # units
    "UnitType",
    "UnitLikeType",
]

__credits__ = ["Astropy"]

##############################################################################
# TYPES
##############################################################################


EllipsisType = type(Ellipsis)
FullPathLike = Union[str, bytes, os.PathLike]


# -------------------------------------
# NumPy types

ArrayLike = Union[float, np.ndarray]  # np.generic isn't compatible


# -------------------------------------
# Astropy types

RepresentationLikeType = Union[BaseRepresentation, str]

CoordinateType = Union[BaseCoordinateFrame, SkyCoord]
"""|Frame| or |SkyCoord|"""

CoordinateLikeType = Union[CoordinateType, str]
"""|Frame| or |SkyCoord| or `str`"""

FrameLikeType = Union[BaseCoordinateFrame, str]
"""|Frame| or str"""

UnitType = Union[u.Unit, u.IrreducibleUnit, u.UnitBase, u.FunctionUnitBase]
"""|Unit| or :class:`~astropy.units.FunctionUnitBase`"""

UnitLikeType = Union[UnitType, str]
"""|Unit| or :class:`~astropy.units.FunctionUnitBase` or str"""


class HasFrame(Protocol):
    @property
    def frame(self) -> BaseCoordinateFrame:
        ...