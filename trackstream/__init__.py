# -*- coding: utf-8 -*-
# see LICENSE.rst

"""Construct Stream Tracks."""

__author__ = "Nathaniel Starkman"
__copyright__ = "Copyright 2020"

__all__ = [
    # modules
    "examples",
    # classes
    "Stream",
    "TrackStream",
    "StreamTrack",
    # instances
    "conf",
]


##############################################################################
# IMPORTS

# keep this content at the top.
from ._astropy_init import *  # noqa: F401, F403  # isort:skip

# LOCAL
from . import example_data as examples
from .config import conf
from .core import StreamTrack, TrackStream
from .stream import Stream
