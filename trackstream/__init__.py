# -*- coding: utf-8 -*-
# see LICENSE.rst

"""Construct Stream Tracks."""

__author__ = "Nathaniel Starkman"
__copyright__ = "Copyright 2020"

__all__ = [
    # modules
    "examples",
]


##############################################################################
# IMPORTS

# Packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *  # noqa: F401, F403  # isort:skip

# ----------------------------------------------------------------------------

# PROJECT-SPECIFIC
from . import core
from . import example_data as examples
from .core import *  # noqa: F401, F403

# ALL
__all__ += core.__all__

##############################################################################
# END
