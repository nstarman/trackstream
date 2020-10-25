# -*- coding: utf-8 -*-

"""Data Conversions."""

__all__ = ["convert_Table_to_ndarray", "convert_Frame_to_ndarray"]


##############################################################################
# IMPORTS

# BUILT-IN

# THIRD PARTY

# THIRD PARTY
import numpy as np
from utilipy.utils.typing import TableType

# PROJECT-SPECIFIC


##############################################################################
# PARAMETERS


##############################################################################
# CODE
##############################################################################


def convert_Table_to_ndarray(data: TableType, **kw) -> np.ndarray:
    """Transform Astropy Table to array.

    The command ``table.as_array()``

    Parameters
    ----------
    data : `~astropy.table.Table`

    Returns
    -------
    unstructured : `~numpy.ndarray`
       Unstructured array with one more dimension.

    Other Parameters
    ----------------
    dtype : dtype, optional, keyword only
       The dtype of the output unstructured array.
    copy : bool, optional, keyword only
        See copy argument to `ndarray.astype`. If true, always return a copy.
        If false, and `dtype` requirements are satisfied, a view is returned.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        See casting argument of `ndarray.astype`. Controls what kind of data
        casting may occur.

    """
    array: np.ndarray = np.lib.recfunctions.structured_to_unstructured(
        data.as_array(),  # 2) converts to numpy recarray
        dtype=kw.get("dtype", None),
        copy=kw.get("copy", False),
        casting=kw.get("casting", "unsafe"),
    )

    return array


# /def


# -------------------------------------------------------------------


def convert_Frame_to_ndarray(data: TableType, **kw) -> np.ndarray:
    """Transform Astropy Frame to array.

    The frame command ``frame.as_array()``

    Parameters
    ----------
    data : `~astropy.coordinates.BaseCoordinateFrame`

    Returns
    -------
    unstructured : `~numpy.ndarray`
       Unstructured array with one more dimension.

    Other Parameters
    ----------------
    dtype : dtype, optional, keyword only
       The dtype of the output unstructured array.
    copy : bool, optional, keyword only
        See copy argument to `ndarray.astype`. If true, always return a copy.
        If false, and `dtype` requirements are satisfied, a view is returned.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        See casting argument of `ndarray.astype`. Controls what kind of data
        casting may occur.

    """
    array: np.ndarray = np.lib.recfunctions.structured_to_unstructured(
        data._values,  # 2) converts to numpy heterogeneous arraay
        dtype=kw.get("dtype", None),
        copy=kw.get("copy", False),
        casting=kw.get("casting", "unsafe"),
    )

    return array


# /def


##############################################################################
# END
