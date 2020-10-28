# -*- coding: utf-8 -*-

"""Testing :mod:`~trackstream.utils.spline`."""

__all__ = ["Test_InterpolatedUnivariateSplinewithUnits"]


##############################################################################
# IMPORTS

# THIRD PARTY
import astropy.units as u
import numpy as np
import pytest
from astropy.tests.helper import assert_quantity_allclose
from astropy.utils.decorators import format_doc

# PROJECT-SPECIFIC
from trackstream.tests.helper import BaseClassDependentTests
from trackstream.utils import spline
from trackstream.utils.spline import (
    InterpolatedUnivariateSplinewithUnits as IUSU,
)

##############################################################################
# TESTS
##############################################################################


@format_doc(
    None,
    package="trackstream.utils.spline",
    klass="InterpolatedUnivariateSplinewithUnits",
)
class Test_InterpolatedUnivariateSplinewithUnits(
    BaseClassDependentTests, klass=spline.InterpolatedUnivariateSplinewithUnits
):
    """Test :class:`~{package}.{klass}`."""

    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        num = 40

        cls.x = np.linspace(0, 180, num=num) * u.deg
        cls.y = np.linspace(0, 10, num=num) * u.m
        cls.w = np.random.rand(num)
        cls.bbox = [0 * u.deg, 180 * u.deg]

        cls.interps = dict(
            basic=IUSU(cls.x, cls.y),
            weighted=IUSU(cls.x, cls.y, w=cls.w),
            bbox=IUSU(cls.x, cls.y, bbox=cls.bbox),
        )

    # /def

    @classmethod
    def teardown_class(cls):
        """Tear-down fixtures for testing."""
        pass

    # /def

    # -------------------------------

    def test_fail_init(self):
        """Test initialization."""
        with pytest.raises(u.UnitConversionError):

            bad_unit = self.x.unit / u.m
            IUSU(self.x, self.y, bbox=[0 * bad_unit, 180 * bad_unit])

        # /with

    # /def

    def test_call(self):
        """Test call method."""
        for name, interp in self.interps.items():
            y = interp(self.x)

            assert_quantity_allclose(y, self.y, atol=1e-16 * y.unit), name

        # /for

    # /def


# /class


# -------------------------------------------------------------------


##############################################################################
# END
