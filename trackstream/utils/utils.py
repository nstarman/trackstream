# -*- coding: utf-8 -*-
# see LICENSE.rst

"""Utilities.

This sub-module is destined for common non-package specific utility functions.
"""


__all__ = ["find_closest_point", "set_starting_point"]


###############################################################################
# IMPORTS

# STDLIB
from typing import Any, Callable, Optional, TypeVar, cast

# THIRD PARTY
import astropy.units as u
import numpy as np

# LOCAL
from trackstream._type_hints import CoordinateType, DummyAttribute

##############################################################################
# CODE
##############################################################################


def find_closest_point(data: CoordinateType, to_point: CoordinateType) -> CoordinateType:
    """Find closest point, on sky if `to_point` is on-sky.

    Parameters
    ----------
    data : |SkyCoord| or |Frame| instance
    to_point : |SkyCoord| or |Frame| instance

    Returns
    -------
    start_point : Sequence
        Shape (# features, ). Point in `data` nearest `near_point` in KDTree
    start_ind : int
        Index into `data` for the `start_point`
        If `return_ind` == True

    """
    # Do we want the closes point in 3D or on-sky?
    if to_point.spherical.distance.unit == u.one:  # on-sky
        seps = data.separation(to_point)
    else:
        seps = data.separation_3d(to_point)

    start_ind = np.argmin(seps)
    start_point = data[start_ind]

    return start_point, start_ind


def set_starting_point(data: CoordinateType, start_ind: int) -> CoordinateType:
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
    new_order = np.array([start_ind, *order])

    return data[new_order]  # return reordered data


R = TypeVar("R")


def abstract_attribute(obj: Optional[Callable[[Any], R]] = None) -> R:
    # https://stackoverflow.com/a/50381071
    _obj = cast(Any, obj)
    if obj is None:
        _obj = DummyAttribute()
    _obj.__is_abstract_attribute__ = True
    return cast(R, _obj)
