# -*- coding: utf-8 -*-

"""Common code, safe for use in sub-packages."""

__all__ = [
    # hits for types
    "ICRSType",
    "QTableType",
    "CoordinateType",
    "QuantityType",
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

ICRSType = T.TypeVar("ICRSType", bound=coord.ICRS)
QTableType = T.TypeVar("QTableType", bound=QTable)
CoordinateType = T.TypeVar("CoordinateType", bound=coord.BaseCoordinateFrame)
QuantityType = T.TypeVar("QuantityType", bound=u.Quantity)


##############################################################################
# CODE
##############################################################################


# -------------------------------------------------------------------


##############################################################################
# END
