# -*- coding: utf-8 -*-

"""Load Example Palomar 5 Data."""

__credits__ = ["Rodrigo Ibata"]


__all__ = [
    "get_stream_data",
]


##############################################################################
# IMPORTS

import os.path

import astropy.coordinates as coord
from astropy.table import QTable
from astropy.utils.data import get_pkg_data_filename

from astronat.utils.table import QTableList

##############################################################################
# CODE
##############################################################################


def _load_data() -> QTable:

    path = get_pkg_data_filename(
        os.path.join("data", "IbataEtAl2017", "vizier.asdf"),
        package="trackstream",
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
