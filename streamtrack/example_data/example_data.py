# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------
#
# TITLE   :
# AUTHOR  :
# PROJECT :
#
# ----------------------------------------------------------------------------

"""Load Example Palomar 5 Data."""

__credits__ = ["Rodrigo Ibata"]


__all__ = [
    "stream_data",
]


##############################################################################
# IMPORTS

# BUILT-IN

import os.path


# THIRD PARTY

from astronat.utils.table import QTableList

import astropy.coordinates as coord
from astropy.utils.data import get_pkg_data_filename
from astropy.table import QTable


# PROJECT-SPECIFIC


##############################################################################
# PARAMETERS


##############################################################################
# CODE
##############################################################################


def _load_data() -> QTable:

    path = get_pkg_data_filename(
        os.path.join("data", "IbataEtAl2017", "vizier.asdf"),
        package="streamtrack",
    )

    data = QTableList.read(path)["table2"]

    return data


# /def


# -------------------------------------------------------------------


def get_stream_data(threshold: float = 0.7) -> coord.SkyCoord:
    """Stream data as `~astropy.coordinates.SkyCoord`."""
    full_data = _load_data()

    sub_data = full_data[["ra", "dec"]][full_data["PMemb"] > threshold]

    data = coord.SkyCoord.guess_from_table(sub_data)

    return data


# /def


##############################################################################
# END
