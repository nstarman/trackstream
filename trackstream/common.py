# -*- coding: utf-8 -*-

"""Common code, safe for use in sub-packages."""

__all__ = [
    # hits for types
    "QuantityType",
    "QTableType",
    "ICRSType",
    "CoordinateType",
    "RepresentationType",
]


##############################################################################
# IMPORTS

# BUILT-IN
import typing as T

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
from astropy.table import QTable

##############################################################################
# PARAMETERS

QuantityType = T.TypeVar("QuantityType", bound=u.Quantity)
QTableType = T.TypeVar("QTableType", bound=QTable)
ICRSType = T.TypeVar("ICRSType", bound=coord.ICRS)
CoordinateType = T.TypeVar(
    "CoordinateFrameType", bound=coord.BaseCoordinateFrame
)
RepresentationType = T.TypeVar(
    "RepresentationType", bound=coord.BaseRepresentation
)


##############################################################################
# CODE
##############################################################################


# -------------------------------------------------------------------


##############################################################################
# END
