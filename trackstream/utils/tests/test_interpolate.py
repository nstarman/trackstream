# -*- coding: utf-8 -*-

"""Testing :mod:`~trackstream.utils.interpolate`."""

__all__ = [
    "Test_UnivariateSplinewithUnits",
    "Test_InterpolatedUnivariateSplinewithUnits",
    "Test_LSQUnivariateSplinewithUnits",
]


##############################################################################
# IMPORTS

# THIRD PARTY
import astropy.units as u
import numpy as np
import pytest
from astropy.tests.helper import assert_quantity_allclose

# LOCAL
from trackstream.tests.helper import BaseClassDependentTests
from trackstream.utils import interpolate

##############################################################################
# TESTS
##############################################################################


class Test_UnivariateSplinewithUnits(
    BaseClassDependentTests,
    klass=interpolate.UnivariateSplinewithUnits,
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
        cls.extra_args = extra_args = dict(
            s=None,
            k=3,
            ext=0,
            check_finite=False,
        )

        cls.spls = dict(
            basic=cls.klass(
                cls.x, cls.y, w=None, bbox=[None] * 2, **extra_args
            ),
            weight=cls.klass(
                cls.x, cls.y, w=cls.w, bbox=[None] * 2, **extra_args
            ),
            bbox=cls.klass(cls.x, cls.y, w=None, bbox=cls.bbox, **extra_args),
        )

    # /def

    # -------------------------------

    def test_fail_init(self):
        """Test a failed initialization b/c wrong units."""
        with pytest.raises(u.UnitConversionError):
            bad_unit = self.x.unit / u.m
            self.klass(self.x, self.y, bbox=[0 * bad_unit, 180 * bad_unit])

    # /def

    def test_call(self):
        """Test call method."""
        for spl in self.spls.values():
            y = spl(self.x)  # evaluate spline
            assert_quantity_allclose(y, self.y, atol=1e-13 * y.unit)

        # /for

    # /def

    def test_get_knots(self):
        """Test method ``get_knots``."""
        for name, spl in self.spls.items():
            knots = spl.get_knots()

            assert knots.unit == self.x.unit, name

    # /def

    def test_get_coeffs(self):
        """Test method ``get_coeffs``."""
        for name, spl in self.spls.items():
            coeffs = spl.get_coeffs()

            assert coeffs.unit == self.y.unit, name

    # /def

    def test_get_residual(self):
        """Test method ``get_residual``."""
        for name, spl in self.spls.items():
            residual = spl.get_residual()

            assert residual.unit == self.y.unit, name

    # /def

    def test_integral(self):
        """Test method ``integral``."""
        for name, spl in self.spls.items():
            integral = spl.integral(self.x[0], self.x[-1])

            assert integral.unit == self.x.unit * self.y.unit

    # /def

    def test_derivatives(self):
        """Test method ``derivatives``."""
        for name, spl in self.spls.items():
            derivatives = spl.derivatives(self.x[3])

            assert derivatives[0].unit == self.y.unit

    # /def

    def test_roots(self):
        """Test method ``roots``."""
        for name, spl in self.spls.items():
            roots = spl.roots()

            assert roots.unit == self.x.unit

    # /def

    def test_derivative(self):
        """Test method ``derivative``."""
        for name, spl in self.spls.items():
            deriv = spl.derivative(n=2)

            assert deriv._xunit == self.x.unit
            assert deriv._yunit == self.y.unit / self.x.unit ** 2

    # /def

    def test_antiderivative(self):
        """Test method ``antiderivative``."""
        for name, spl in self.spls.items():
            antideriv = spl.antiderivative(n=2)

            assert antideriv._xunit == self.x.unit
            assert antideriv._yunit == self.y.unit * self.x.unit ** 2

    # /def


# /class


# -------------------------------------------------------------------


class Test_InterpolatedUnivariateSplinewithUnits(
    Test_UnivariateSplinewithUnits,
    klass=interpolate.InterpolatedUnivariateSplinewithUnits,
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

        cls.spls = dict(
            basic=cls.klass(
                cls.x, cls.y, w=None, bbox=[None] * 2, **extra_args
            ),
            weight=cls.klass(
                cls.x, cls.y, w=cls.w, bbox=[None] * 2, **extra_args
            ),
            bbox=cls.klass(cls.x, cls.y, w=None, bbox=cls.bbox, **extra_args),
        )

    # /def

    # -------------------------------


# /class


# -------------------------------------------------------------------


class Test_LSQUnivariateSplinewithUnits(
    Test_UnivariateSplinewithUnits,
    klass=interpolate.LSQUnivariateSplinewithUnits,
):
    """Test :class:`~{package}.{klass}`."""

    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        num = 40

        cls.x = np.linspace(0, 6, num=num) * u.deg
        cls.y = (np.exp(-(cls.x.value ** 2)) + 0.1) * u.m
        cls.w = np.random.rand(num)
        cls.bbox = [0 * u.deg, 180 * u.deg]
        cls.extra_args = extra_args = dict(k=3, ext=0, check_finite=False)

        klass = interpolate.InterpolatedUnivariateSplinewithUnits
        spl = klass(cls.x, cls.y, w=None, bbox=[None] * 2, **extra_args)
        cls.t = spl.get_knots().value[1:-1]

        cls.spls = dict(
            basic=cls.klass(
                cls.x, cls.y, cls.t, w=None, bbox=[None] * 2, **extra_args
            ),
            weight=cls.klass(
                cls.x, cls.y, cls.t, w=cls.w, bbox=[None] * 2, **extra_args
            ),
            bbox=cls.klass(
                cls.x, cls.y, cls.t, w=None, bbox=cls.bbox, **extra_args
            ),
        )

    # /def

    # -------------------------------

    def test_fail_init(self):
        """Test a failed initialization b/c wrong units."""
        with pytest.raises(u.UnitConversionError):
            bad_unit = self.x.unit / u.m
            self.klass(
                self.x,
                self.y,
                self.t,
                bbox=[0 * bad_unit, 180 * bad_unit],
            )

    # /def


# /class


# -------------------------------------------------------------------


##############################################################################
# END
