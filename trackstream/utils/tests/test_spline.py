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
        cls.extra_args = extra_args = dict(k=3, ext=0, check_finite=False)

        cls.interps = dict(
            basic=IUSU(cls.x, cls.y, w=None, bbox=[None] * 2, **extra_args),
            weight=IUSU(cls.x, cls.y, w=cls.w, bbox=[None] * 2, **extra_args),
            bbox=IUSU(cls.x, cls.y, w=None, bbox=cls.bbox, **extra_args),
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

            assert_quantity_allclose(y, self.y, atol=1e-15 * y.unit), name

        # /for

    # /def

    def test_get_knots(self):
        """Test method ``get_knots``."""
        for name, interp in self.interps.items():
            knots = interp.get_knots()

            assert knots.unit == self.x.unit

    # /def

    def test_get_coeffs(self):
        """Test method ``get_coeffs``."""
        for name, interp in self.interps.items():
            coeffs = interp.get_coeffs()
            assert not hasattr(coeffs, "unit")

    # /def

    def test_get_residual(self):
        """Test method ``get_residual``."""
        for name, interp in self.interps.items():
            residual = interp.get_residual()

            assert residual.unit == self.y.unit

    # /def

    def test_integral(self):
        """Test method ``integral``."""
        for name, interp in self.interps.items():
            integral = interp.integral(self.x[0], self.x[-1])

            assert integral.unit == self.y.unit

    # /def

    def test_derivatives(self):
        """Test method ``derivatives``."""
        for name, interp in self.interps.items():
            derivatives = interp.derivatives(self.x[3])

            assert derivatives[0].unit == self.y.unit

    # /def

    def test_roots(self):
        """Test method ``roots``."""
        for name, interp in self.interps.items():
            roots = interp.roots()

            assert roots.unit == self.x.unit

    # /def

    def test_derivative(self):
        """Test method ``derivative``."""
        for name, interp in self.interps.items():
            with pytest.raises(NotImplementedError):
                interp.derivative(n=2)

    # /def

    def test_antiderivative(self):
        """Test method ``antiderivative``."""
        for name, interp in self.interps.items():
            with pytest.raises(NotImplementedError):
                interp.antiderivative(n=2)

    # /def


# /class


# -------------------------------------------------------------------


##############################################################################
# END
