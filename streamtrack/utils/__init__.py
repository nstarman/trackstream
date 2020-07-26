# -*- coding: utf-8 -*-
# see LICENSE.rst

"""Utilities.

This sub-module is destined for common non-package specific utility functions.

"""


__all__ = [
    # modules
    "core",
    "coordinates",
    # functions
    "cartesian_to_spherical",
    "reference_to_skyoffset_matrix",
    "get_transform_matrix",
    "p2p_distance_cartesian",
    "p2p_distance_spherical",
]


##############################################################################
# IMPORTS

# PROJECT-SPECIFIC

from .core import p2p_distance_cartesian, p2p_distance_spherical

from .coordinates import (
    cartesian_to_spherical,
    reference_to_skyoffset_matrix,
    get_transform_matrix,
)

from . import core, coordinates

##############################################################################
# END
