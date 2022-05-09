# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Set up package."""


##############################################################################
# IMPORTS

from __future__ import absolute_import

try:
    # THIRD PARTY
    import tqdm  # noqa: F401
except ImportError:
    HAS_TQDM = False
else:
    HAS_TQDM = True


__all__ = ["HAS_TQDM"]
