# see LICENSE.rst

"""Construct Stream Tracks."""

##############################################################################
# IMPORTS

from __future__ import annotations

# LOCAL
from .fit import FitterStreamArmTrack

# from . import example_data as examples
from .stream import Stream

# keep this content at the top.
from ._astropy_init import *  # noqa: F401, F403  # isort:skip

__all__ = ["Stream", "FitterStreamArmTrack"]

__author__ = "Nathaniel Starkman"
__copyright__ = "Copyright 2020"
