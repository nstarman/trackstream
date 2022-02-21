# -*- coding: utf-8 -*-
# see LICENSE.rst

"""Utilities.

This sub-module is destined for common non-package specific utility functions.

"""


__all__ = ["find_closest_point", "set_starting_point"]


###############################################################################
# IMPORTS

# STDLIB
import typing as T
from collections.abc import Sequence

# THIRD PARTY
import numpy as np
from astropy.coordinates import BaseRepresentation, CartesianRepresentation

# LOCAL
from trackstream._type_hints import CoordinateType

##############################################################################
# PARAMETERS

DataType = T.Union[CoordinateType, BaseRepresentation]

##############################################################################
# CODE
##############################################################################


def find_closest_point(
    data: DataType,
    near_point: T.Union[BaseRepresentation, T.Sequence, np.ndarray],
):
    """Find starting point.

    Parameters
    ----------
    data : |Representation| or |Frame| instance
        Shape (# measurements, # features).
        Must be transformable to Cartesian coordinates.
    near_point : Sequence
        Shape (1, # features)
        If passing an array, can reshape with ``.reshape(1, -1)``

    Returns
    -------
    start_point : Sequence
        Shape (# features, ). Point in `data` nearest `near_point` in KDTree
    start_ind : int
        Index into `data` for the `start_point`
        If `return_ind` == True

    """
    if isinstance(near_point, (Sequence, np.ndarray)):
        near_point = CartesianRepresentation(near_point)
    else:
        near_point = near_point.represent_as(CartesianRepresentation)

    data = data.represent_as(CartesianRepresentation)

    start_ind = np.argmin((data - near_point).norm())
    start_point = data[start_ind]

    return start_point, start_ind


def set_starting_point(data: DataType, start_ind: int):
    """Reorder data to set starting index at row 0.

    Parameters
    ----------
    data : |Representation| or |Frame| instance
    start_ind : int

    Returns
    -------
    `data`
        Re-ordered.
    """
    # index order array
    order = list(range(len(data)))
    del order[start_ind]
    order = np.array([start_ind, *order])

    return data[order]  # return reordered data


##############################################################################
# END
