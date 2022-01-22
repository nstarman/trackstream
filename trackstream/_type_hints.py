# -*- coding: utf-8 -*-

"""Type hints.

This project extensively uses :mod:`~typing` hints.
Note that this is not (necessarily) static typing.


**TypeVar**

Most of the types are :class:`~typing.TypeVar` with a standard format: for an
object X, the variable name and TypeVar name are "{X}Type" and the TypeVar is
bound to X such that all subclasses of X permit the same type hint.

As a trivial example,

    >>> import typing as T
    >>> IntType = T.TypeVar("Int", bound=int)

``IntType`` works on any subclass (inclusive) of int.

"""

__all__ = [
    "ArrayLike",
    # coordinates
    "RepresentationOrDifferentialType",
    "FrameType",
    "CoordinateType",
    "FrameLikeType",
    # units
    "UnitType",
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

ArrayLike = T.Union[float, np.generic, np.ndarray]


# -------------------------------------
# Astropy types

RepLikeType = T.Union[coord.BaseRepresentation, str]

FrameType = T.TypeVar("CoordinateFrame", bound=coord.BaseCoordinateFrame)
"""|Frame|"""

CoordinateType = T.Union[FrameType, coord.SkyCoord]
"""|Frame| or |SkyCoord|"""

FrameLikeType = T.Union[CoordinateType, str]
"""|Frame| or |SkyCoord| or `str`"""

UnitType = T.Union[u.UnitBase, u.FunctionUnitBase]
"""|Unit| or :class:`~astropy.units.FunctionUnitBase`"""

##############################################################################
# END
