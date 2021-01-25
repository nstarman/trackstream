# -*- coding: utf-8 -*-
# see LICENSE.rst

"""Tests for :mod:`~trackstream.preprocess`.

An Astropy test-runner is also provided.

"""

__all__ = [
    # modules
    "rotated_frame_tests",
    # instances
    "test",
]


##############################################################################
# IMPORTS

# BUILT-IN
from pathlib import Path

# THIRD PARTY
from astropy.tests.runner import TestRunner

# PROJECT-SPECIFIC
from . import test_rotated_frame as rotated_frame_tests

##############################################################################
# PARAMETERS
##############################################################################

test = TestRunner.make_test_runner_in(Path(__file__).parent.parent)

##############################################################################
# END
