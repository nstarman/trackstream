# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Set up package."""


##############################################################################
# IMPORTS

from __future__ import absolute_import

try:
    # THIRD PARTY
    from minisom import MiniSom  # noqa: F401
except ImportError:
    HAS_MINISOM = False
else:
    HAS_MINISOM = True


try:
    # THIRD PARTY
    from filterpy import kalman  # noqa: F401
except ImportError:
    HAS_FILTERPY = False
else:
    HAS_FILTERPY = True


__all__ = ["HAS_MINISOM", "HAS_FILTERPY"]

##############################################################################
# END
