# -*- coding: utf-8 -*-
# see LICENSE.rst

"""Unit Tests for `~trackstream.utils`."""


__all__ = [
    "interpolation_tests",
    "interpolated_coordinates_test",
    # instance
    "test",
]


##############################################################################
# IMPORTS

# STDLIB
from pathlib import Path

# THIRD PARTY
from astropy.tests.runner import TestRunner

# LOCAL
from . import test_interpolate as interpolation_tests
from . import test_interpolated_coordinates as interpolated_coordinates_test

##############################################################################
# TESTS
##############################################################################

test = TestRunner.make_test_runner_in(Path(__file__).parent.parent)

##############################################################################
# END
