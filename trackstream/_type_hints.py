# -*- coding: utf-8 -*-

"""Type hints.

This project extensively uses :mod:`~typing` hints.
Note that this is not (necessarily) static typing.
"""

__all__ = [
    "ArrayLike",
    # coordinates
    "FrameType",
    "CoordinateType",
    "FrameLikeType",
    # units
    "UnitType",
    "UnitLikeType",
]

__credits__ = ["Astropy"]


##############################################################################
# IMPORTS

# STDLIB
from typing import TypeVar, Union

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import numpy as np

##############################################################################
# TYPES
##############################################################################

# -------------------------------------
# NumPy types

ArrayLike = Union[float, np.ndarray]  # np.generic isn't compatible


# -------------------------------------
# Astropy types

RepLikeType = Union[coord.BaseRepresentation, str]

FrameType = TypeVar("FrameType", bound=coord.BaseCoordinateFrame)
"""|Frame|"""

CoordinateType = Union[FrameType, coord.SkyCoord]
"""|Frame| or |SkyCoord|"""

FrameLikeType = Union[CoordinateType, str]
"""|Frame| or |SkyCoord| or `str`"""

UnitType = Union[u.UnitBase, u.FunctionUnitBase]
"""|Unit| or :class:`~astropy.units.FunctionUnitBase`"""

UnitLikeType = Union[UnitType, str]
"""|Unit| or :class:`~astropy.units.FunctionUnitBase` or str"""

##############################################################################
# END
