# -*- coding: utf-8 -*-

"""Type hints.

This project extensively uses :mod:`~typing` hints.
Note that this is not (necessarily) static typing.
"""

__all__ = [
    "EllipsisType",
    "FullPathLike",
    "ArrayLike",
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
import os
from typing import Union

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import numpy as np

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

RepresentationLikeType = Union[coord.BaseRepresentation, str]

CoordinateType = Union[coord.BaseCoordinateFrame, coord.SkyCoord]
"""|Frame| or |SkyCoord|"""

FrameLikeType = Union[CoordinateType, str]
"""|Frame| or |SkyCoord| or `str`"""

UnitType = Union[u.Unit, u.IrreducibleUnit, u.UnitBase, u.FunctionUnitBase]
"""|Unit| or :class:`~astropy.units.FunctionUnitBase`"""

UnitLikeType = Union[UnitType, str]
"""|Unit| or :class:`~astropy.units.FunctionUnitBase` or str"""


class DummyAttribute:
    __is_abstract_attribute__: bool
