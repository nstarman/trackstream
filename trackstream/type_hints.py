# -*- coding: utf-8 -*-

"""Type hints.

This project extensively uses `~typing` hints.
Note that this is NOT static typing.


**TypeVar**

Most of the types are :mod:`~typing.TypeVar` with a standard format: for an
object X, the variable name and TypeVar name are "{X}Type" and the TypeVar is
bound to X such that all subclasses of X permit the same type hint.

As a trivial example,

    >>> import typing as T
    >>> IntType = T.TypeVar("Int", bound=int)

``IntType`` works on any subclass (inclusive) of int.

"""

__all__ = [
    "NoneType",
    "EllipsisType",
    "UnitType",
    "UnitLikeType",
    "QuantityType",
    "QuantityLikeType",
    "FrameType",
    "SkyCoordType",
    "CoordinateType",
    "FrameLikeType",
    "RepresentationType",
    "DifferentialType",
    "TableType",
    "ModelType",
]

__credits__ = ["Astropy"]


##############################################################################
# IMPORTS

# BUILT-IN
import typing as T

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
from astropy.modeling import Model
from astropy.table import Table

##############################################################################
# TYPES
##############################################################################

NoneType = type(None)
EllipsisType = type(Ellipsis)

#####################################################################
# Astropy types

# -----------------
# units

UnitType = T.Union[
    T.TypeVar("Unit", bound=u.UnitBase),
    T.TypeVar("FunctionUnit", bound=u.FunctionUnitBase),
]
UnitLikeType = T.Union[UnitType, str]

QuantityType = T.TypeVar("Quantity", bound=u.Quantity)
QuantityLikeType = T.Union[QuantityType, str]

# -----------------
# coordinates

RepresentationType = T.TypeVar(
    "BaseRepresentation",
    bound=coord.BaseRepresentation,
)
DifferentialType = T.TypeVar("BaseDifferential", bound=coord.BaseDifferential)

FrameType = T.TypeVar("CoordinateFrame", bound=coord.BaseCoordinateFrame)
SkyCoordType = T.TypeVar("SkyCoord", bound=coord.SkyCoord)
CoordinateType = T.Union[FrameType, SkyCoordType]

FrameLikeType = T.Union[CoordinateType, str]

# -----------------
# modeling

ModelType = T.TypeVar("Model", bound=Model)

# -----------------
# table

TableType = T.TypeVar("Table", bound=Table)

##############################################################################
# END
