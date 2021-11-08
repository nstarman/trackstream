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
    # Astropy types
    # coordinates
    "RepresentationOrDifferentialType",
    "RepresentationType",
    "DifferentialType",
    "FrameType",
    "SkyCoordType",
    "CoordinateType",
    "PositionType",
    "GenericPositionType",
    "FrameLikeType",
    # tables
    "TableType",
    "QTableType",
    # units
    "UnitType",
    "UnitLikeType",
    "QuantityType",
    "QuantityLikeType",
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
from astropy import table

##############################################################################
# TYPES
##############################################################################

# -------------------------------------
# NumPy types

ArrayLike = T.Union[float, np.generic, np.ndarray]


# -------------------------------------
# Astropy types

# -----------------
# coordinates

RepresentationOrDifferentialType = T.TypeVar(
    "BaseRepresentationOrDifferential",
    bound=coord.BaseRepresentationOrDifferential,
)
"""|RepresentationOrDifferential|"""

RepresentationType = T.TypeVar(
    "BaseRepresentation",
    bound=coord.BaseRepresentation,
)
"""|Representation|"""

DifferentialType = T.TypeVar("BaseDifferential", bound=coord.BaseDifferential)
"""|Differential|"""

FrameType = T.TypeVar("CoordinateFrame", bound=coord.BaseCoordinateFrame)
"""|Frame|"""

SkyCoordType = T.TypeVar("SkyCoord", bound=coord.SkyCoord)
"""|SkyCoord|"""

CoordinateType = T.Union[FrameType, SkyCoordType]
"""|Frame| or |SkyCoord|"""

PositionType = T.Union[RepresentationType, CoordinateType]
"""|BaseRepresentation|, |Frame|, or |SkyCoord|"""

GenericPositionType = T.Union[RepresentationOrDifferentialType, CoordinateType]
"""|BaseRepresentationOrDifferential|, |Frame|, or |SkyCoord|"""

FrameLikeType = T.Union[CoordinateType, str]
"""|Frame| or |SkyCoord| or `str`"""

# -----------------
# table

TableType = T.TypeVar("Table", bound=table.Table)
"""|Table|"""

QTableType = T.TypeVar("QTable", bound=table.QTable)
"""|QTable|"""

# -----------------
# units

UnitType = T.Union[
    T.TypeVar("Unit", bound=u.UnitBase),
    T.TypeVar("FunctionUnit", bound=u.FunctionUnitBase),
]
"""|Unit| or :class:`~astropy.units.FunctionUnitBase`"""

UnitLikeType = T.Union[UnitType, str]
"""|Unit|, :class:`~astropy.units.FunctionUnitBase`, or `str`"""

QuantityType = T.TypeVar("Quantity", bound=u.Quantity)
"""|Quantity|"""

QuantityLikeType = T.Union[QuantityType, str]
"""|Quantity| or `str`"""

##############################################################################
# END
