# -*- coding: utf-8 -*-
# see LICENSE.rst

"""Construct Stream Tracks."""

__author__ = "Nathaniel Starkman"
__copyright__ = "Copyright 2020"

__all__ = [
    # modules
    "examples",
    "preprocess",
    # classes
    "Stream",
    "TrackStream",
    "StreamTrack",
    # instances
    "conf",
]


##############################################################################
# IMPORTS

# Packages may add whatever they like to this file, but
# should keep this content at the top.
from ._astropy_init import *  # noqa: F401, F403  # isort:skip

# LOCAL
from . import core
from . import example_data as examples
from . import preprocess
from .config import conf
from .core import StreamTrack, TrackStream
from .stream import Stream

##############################################################################
# END
