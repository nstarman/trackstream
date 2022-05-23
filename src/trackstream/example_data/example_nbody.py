# -*- coding: utf-8 -*-

"""Example NBody Data."""

__author__ = "Nathaniel Starkman"
__credits__ = ["Jeremy Webb for N-Body Data"]


__all__ = ["get_nbody", "get_nbody_array"]


##############################################################################
# IMPORTS

# THIRD PARTY
import numpy as np
from astropy.coordinates import BaseCoordinateFrame, Galactocentric
from astropy.table import QTable
from astropy.utils.data import get_pkg_data_filename

##############################################################################
# CODE
##############################################################################


def _load_nbody() -> QTable:
    """Get Palomar 5 at pericenter Nbody Data.

    TODO move to ZENODO and download to cache instead.

    Returns
    -------
    data : `~astropy.table.QTable`

        File in "data/00590_peri.dat"
    """
    # Filename
    fname: str = get_pkg_data_filename("data/00590_peri.dat", package="trackstream")
    # Read
    data: QTable = QTable.read(fname, format="ascii.ecsv")
    return data


def get_nbody(subsample: slice = slice(100, None, 400)) -> BaseCoordinateFrame:
    """Get NBody.

    Parameters
    ----------
    subsample : slice

    Returns
    -------
    `~astropy.coordinates.Galactocentric`
    """
    full_data: QTable = _load_nbody()
    sub_data: QTable = full_data[subsample][["x", "y", "z"]]

    data: BaseCoordinateFrame = Galactocentric(
        x=sub_data["x"],
        y=sub_data["y"],
        z=sub_data["z"],
    )

    return data


# -------------------------------------------------------------------


def get_nbody_array(subsample: slice = slice(100, None, 400)) -> np.ndarray:
    """Get Palomar 5 at pericenter Nbody Data.

    TODO move to ZENODO and download to cache instead.

    Returns
    -------
    data : `~astropy.table.QTable`
        File location "data/00590_peri.dat". Shape (N, 3)

    """
    full_data: QTable = _load_nbody()

    data: np.ndarray = (
        full_data[subsample][["x", "y", "z"]]  # subsamples full_data`.
        .as_array()  # converts to numpy recarray
        .view("<f8")  # normal, homogeneous dtype ndarray
        .reshape(-1, 3)  # reshapes from flattened array
    )

    return data
