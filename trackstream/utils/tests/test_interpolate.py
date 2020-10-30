# -*- coding: utf-8 -*-

"""Testing :mod:`~trackstream.utils.interp`."""

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
from trackstream.utils import interpolate as interp

##############################################################################
# TESTS
##############################################################################


@format_doc(
    None,
    package="trackstream.utils.interp",
    klass="UnivariateSplinewithUnits",
)
class Test_UnivariateSplinewithUnits(
    BaseClassDependentTests, klass=interp.UnivariateSplinewithUnits
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
            s=None, k=3, ext=0, check_finite=False
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
            self.klass(self.x, self.y, bbox=[0 * bad_unit, 180 * bad_unit])

        # /with

    # /def

    def test_call(self):
        """Test call method."""
        for spl in self.spls.values():
            y = spl(self.x)

            assert_quantity_allclose(y, self.y, atol=1e-15 * y.unit)

        # /for

    # /def

    def test_get_knots(self):
        """Test method ``get_knots``."""
        for name, spl in self.spls.items():
            knots = interp.get_knots()

            assert knots.unit == self.x.unit, name

    # /def

    def test_get_coeffs(self):
        """Test method ``get_coeffs``."""
        for name, spl in self.spls.items():
            coeffs = interp.get_coeffs()

            assert coeffs.unit == self._yunit, name

    # /def

    def test_get_residual(self):
        """Test method ``get_residual``."""
        for name, spl in self.spls.items():
            residual = interp.get_residual()

            assert residual.unit == self.y.unit, name

    # /def

    def test_integral(self):
        """Test method ``integral``."""
        for name, spl in self.spls.items():
            integral = interp.integral(self.x[0], self.x[-1])

            assert integral.unit == self.y.unit

    # /def

    def test_derivatives(self):
        """Test method ``derivatives``."""
        for name, spl in self.spls.items():
            derivatives = interp.derivatives(self.x[3])

            assert derivatives[0].unit == self.y.unit

    # /def

    def test_roots(self):
        """Test method ``roots``."""
        for name, spl in self.spls.items():
            roots = interp.roots()

            assert roots.unit == self.x.unit

    # /def

    def test_derivative(self):
        """Test method ``derivative``."""
        for name, spl in self.spls.items():
            with pytest.raises(NotImplementedError):
                interp.derivative(n=2)

    # /def

    def test_antiderivative(self):
        """Test method ``antiderivative``."""
        for name, spl in self.spls.items():
            with pytest.raises(NotImplementedError):
                interp.antiderivative(n=2)

    # /def


# /class


# -------------------------------------------------------------------


@format_doc(
    None,
    package="trackstream.utils.interp",
    klass="InterpolatedUnivariateSplinewithUnits",
)
class Test_InterpolatedUnivariateSplinewithUnits(
    BaseClassDependentTests, klass=interp.InterpolatedUnivariateSplinewithUnits
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
            self.klass(self.x, self.y, bbox=[0 * bad_unit, 180 * bad_unit])

        # /with

    # /def

    def test_call(self):
        """Test call method."""
        for name, spl in self.spls.items():
            y = interp(self.x)

            assert_quantity_allclose(y, self.y, atol=1e-15 * y.unit), name

        # /for

    # /def

    def test_get_knots(self):
        """Test method ``get_knots``."""
        for name, spl in self.spls.items():
            knots = interp.get_knots()

            assert knots.unit == self.x.unit

    # /def

    def test_get_coeffs(self):
        """Test method ``get_coeffs``."""
        for name, spl in self.spls.items():
            coeffs = interp.get_coeffs()
            assert not hasattr(coeffs, "unit")

    # /def

    def test_get_residual(self):
        """Test method ``get_residual``."""
        for name, spl in self.spls.items():
            residual = interp.get_residual()

            assert residual.unit == self.y.unit

    # /def

    def test_integral(self):
        """Test method ``integral``."""
        for name, spl in self.spls.items():
            integral = interp.integral(self.x[0], self.x[-1])

            assert integral.unit == self.y.unit

    # /def

    def test_derivatives(self):
        """Test method ``derivatives``."""
        for name, spl in self.spls.items():
            derivatives = interp.derivatives(self.x[3])

            assert derivatives[0].unit == self.y.unit

    # /def

    def test_roots(self):
        """Test method ``roots``."""
        for name, spl in self.spls.items():
            roots = interp.roots()

            assert roots.unit == self.x.unit

    # /def

    def test_derivative(self):
        """Test method ``derivative``."""
        for name, spl in self.spls.items():
            with pytest.raises(NotImplementedError):
                interp.derivative(n=2)

    # /def

    def test_antiderivative(self):
        """Test method ``antiderivative``."""
        for name, spl in self.spls.items():
            with pytest.raises(NotImplementedError):
                interp.antiderivative(n=2)

    # /def


# /class


# -------------------------------------------------------------------


##############################################################################
# END
