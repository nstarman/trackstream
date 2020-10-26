# -*- coding: utf-8 -*-

"""Example NBody Data."""

__author__ = "Nathaniel Starkman"
__credits__ = ["Jeremy Webb for N-Body Data"]


__all__ = [
    "get_nbody",
    "get_nbody_array",
]


##############################################################################
# IMPORTS

# BUILT-IN
import os.path

# THIRD PARTY
import astropy.coordinates as coord
import numpy as np
from astropy.table import QTable
from astropy.utils.data import get_pkg_data_filename

# FIRST PARTY
from utilipy.utils.typing import CoordinateType, TableType

##############################################################################
# CODE
##############################################################################


def _load_nbody() -> TableType:
    """Get Palomar 5 at pericenter Nbody Data.

    TODO move to ZENODO and download to cache instead.

    Returns
    -------
    data : `~astropy.table.QTable`

        File in "data/00590_peri.dat"

    """
    fname: str = get_pkg_data_filename(
        os.path.join("data", "00590_peri.dat"), package="trackstream"
    )

    data: TableType = QTable.read(fname, format="ascii.ecsv")

    return data


# /def

# -------------------------------------------------------------------


def get_nbody(subsample=slice(100, None, 400)) -> CoordinateType:
    """Get NBody.

    Parameters
    ----------
    subsample : slice

    Returns
    -------
    `~astropy.coordinates.Galactocentric`

    """
    full_data: TableType = _load_nbody()
    sub_data: TableType = full_data[subsample][["x", "y", "z"]]

    data: CoordinateType = coord.Galactocentric(
        x=sub_data["x"], y=sub_data["y"], z=sub_data["z"]
    )

    return data


# /def


# -------------------------------------------------------------------


def get_nbody_array(subsample=slice(100, None, 400)) -> np.ndarray:
    """Get Palomar 5 at pericenter Nbody Data.

    TODO move to ZENODO and download to cache instead.

    Returns
    -------
    data : `~astropy.table.QTable`
        File location "data/00590_peri.dat". Shape (N, 3)

    """
    full_data: TableType = _load_nbody()

    data: np.ndarray = (
        full_data[subsample][["x", "y", "z"]]  # 1) subsamples full_data`.
        .as_array()  # 2) converts to numpy recarray
        .view("<f8")  # 3)  normal, homogeneous dtype ndarray
        .reshape(-1, 3)  # 5) reshapes from flattened array
    )

    return data


# /def


##############################################################################
# END
