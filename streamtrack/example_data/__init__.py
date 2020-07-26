# -*- coding: utf-8 -*-
# see LICENSE.rst

# ----------------------------------------------------------------------------
#
# TITLE   : Example Data
# PROJECT : Stream Tracker
#
# ----------------------------------------------------------------------------

"""Example Data."""


__all__ = [
    # modules
    "example_coords",
    "example_nbody",
    "example_orbit",
    # functions & classes
    "get_nbody_array",
    "get_nbody",
    "RotatedICRS",
    "get_orbit",
    "make_ordered_orbit_data",
    "make_unordered_orbit_data",
    "make_noisy_orbit_data",
]


##############################################################################
# IMPORTS

# PROJECT-SPECIFIC

from .example_coords import RotatedICRS
from .example_nbody import get_nbody, get_nbody_array
from .example_orbit import (
    get_orbit,
    make_ordered_orbit_data,
    make_unordered_orbit_data,
    make_noisy_orbit_data,
)

from . import example_coords, example_orbit, example_nbody


##############################################################################
# END
