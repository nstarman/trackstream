# -*- coding: utf-8 -*-
# see LICENSE.rst

"""Utilities.

This sub-module is destined for common non-package specific utility functions.
"""


__all__ = [
    "resolve_framelike",
    "reference_to_skyoffset_matrix",
]


##############################################################################
# IMPORTS

# LOCAL
from .coord_utils import reference_to_skyoffset_matrix, resolve_framelike
