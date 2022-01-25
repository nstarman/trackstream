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
import typing as T

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import numpy as np

##############################################################################
# TYPES
##############################################################################

# -------------------------------------
# NumPy types

ArrayLike = T.Union[float, np.ndarray]  # np.generic isn't compatible


# -------------------------------------
# Astropy types

RepLikeType = T.Union[coord.BaseRepresentation, str]

FrameType = T.TypeVar("FrameType", bound=coord.BaseCoordinateFrame)
"""|Frame|"""

CoordinateType = T.Union[FrameType, coord.SkyCoord]
"""|Frame| or |SkyCoord|"""

FrameLikeType = T.Union[CoordinateType, str]
"""|Frame| or |SkyCoord| or `str`"""

UnitType = T.Union[u.UnitBase, u.FunctionUnitBase]
"""|Unit| or :class:`~astropy.units.FunctionUnitBase`"""

UnitLikeType = T.Union[UnitType, str]
"""|Unit| or :class:`~astropy.units.FunctionUnitBase` or str"""

##############################################################################
# END
