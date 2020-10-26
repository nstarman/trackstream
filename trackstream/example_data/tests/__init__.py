# -*- coding: utf-8 -*-
# see LICENSE.rst

"""Tests for `~trackstream.example_data`.

An Astropy test-runner is also provided.

"""

__all__ = [
    # modules
    "example_coords_tests",
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
from . import test_example_coords as example_coords_tests

##############################################################################
# PARAMETERS
##############################################################################

test = TestRunner.make_test_runner_in(Path(__file__).parent.parent)

##############################################################################
# END
