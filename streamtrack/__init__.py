# -*- coding: utf-8 -*-
# see LICENSE.rst

"""Stream Tracker."""

__author__ = "Nathaniel Starkman"
__copyright__ = "Copyright 2020"
# __credits__ = [""]
# __license__ = ""
# __version__ = "0.0.0"
# __maintainer__ = ""
# __email__ = ""
# __status__ = "Production"

__all__ = [
    # modules
    "examples",
    "utils",
    # functions
    "reference_to_skyoffset_matrix",
]


##############################################################################
# IMPORTS

# Packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *  # noqa

# ----------------------------------------------------------------------------

# PROJECT-SPECIFIC

# functions

from .utils import reference_to_skyoffset_matrix

# modules

from . import example_data as examples
from . import utils


##############################################################################
# END
