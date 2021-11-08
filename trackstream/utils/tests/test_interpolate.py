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
from trackstream.utils import interpolate

##############################################################################
# TESTS
##############################################################################


class Test_UnivariateSplinewithUnits:
    """Test UnivariateSplinewithUnits."""

    @pytest.fixture
    def num(self):
        return 40

    @pytest.fixture
    def x(self, num):
        return np.linspace(0, 180, num=num) * u.deg

    @pytest.fixture
    def y(self, num):
        return np.linspace(0, 10, num=num) * u.m

    @pytest.fixture
    def w(self, num):
        return np.random.rand(num)  # TODO? make deterministic?

    @pytest.fixture
    def extra_args(self):
        return dict(s=None, k=3, ext=0, check_finite=False)

    @pytest.fixture
    def bbox(self):
        return [0 * u.deg, 180 * u.deg]

    @pytest.fixture
    def ispline_cls(self):
        return interpolate.UnivariateSplinewithUnits

    @pytest.fixture
    def spls(self, ispline_cls, x, y, w, bbox, extra_args):
        return dict(
            basic=ispline_cls(x, y, w=None, bbox=[None] * 2, **extra_args),
            weight=ispline_cls(x, y, w=w, bbox=[None] * 2, **extra_args),
            bbox=ispline_cls(x, y, w=None, bbox=bbox, **extra_args),
        )

    # -------------------------------

    def test_fail_init(self, ispline_cls, x, y):
        """Test a failed initialization b/c wrong units."""
        with pytest.raises(u.UnitConversionError):
            bad_unit = x.unit / u.m
            ispline_cls(x, y, bbox=[0 * bad_unit, 180 * bad_unit])

    def test_call(self, spls, x, y):
        """Test call method."""
        for spl in spls.values():
            got = spl(x)  # evaluate spline
            assert_quantity_allclose(got, y, atol=1e-13 * y.unit)

        # /for

    def test_get_knots(self, spls, x):
        """Test method ``get_knots``."""
        for name, spl in spls.items():
            knots = spl.get_knots()

            assert knots.unit == x.unit, name

    def test_get_coeffs(self, spls, y):
        """Test method ``get_coeffs``."""
        for name, spl in spls.items():
            coeffs = spl.get_coeffs()

            assert coeffs.unit == y.unit, name

    def test_get_residual(self, spls, y):
        """Test method ``get_residual``."""
        for name, spl in spls.items():
            residual = spl.get_residual()

            assert residual.unit == y.unit, name

    def test_integral(self, spls, x, y):
        """Test method ``integral``."""
        for name, spl in spls.items():
            integral = spl.integral(x[0], x[-1])

            assert integral.unit == x.unit * y.unit

    def test_derivatives(self, spls, x, y):
        """Test method ``derivatives``."""
        for name, spl in spls.items():
            derivatives = spl.derivatives(x[3])

            assert derivatives[0].unit == y.unit

    def test_roots(self, spls, x):
        """Test method ``roots``."""
        for name, spl in spls.items():
            roots = spl.roots()

            assert roots.unit == x.unit

    def test_derivative(self, spls, x, y):
        """Test method ``derivative``."""
        for name, spl in spls.items():
            deriv = spl.derivative(n=2)

            assert deriv._xunit == x.unit
            assert deriv._yunit == y.unit / x.unit ** 2

    def test_antiderivative(self, spls, x, y):
        """Test method ``antiderivative``."""
        for name, spl in spls.items():
            antideriv = spl.antiderivative(n=2)

            assert antideriv._xunit == x.unit
            assert antideriv._yunit == y.unit * x.unit ** 2


# -------------------------------------------------------------------


class Test_InterpolatedUnivariateSplinewithUnits(Test_UnivariateSplinewithUnits):
    """Test UnivariateSplinewithUnits."""

    @pytest.fixture
    def extra_args(self):
        return dict(k=3, ext=0, check_finite=False)

    @pytest.fixture
    def ispline_cls(self):
        return interpolate.InterpolatedUnivariateSplinewithUnits


# -------------------------------------------------------------------


class Test_LSQUnivariateSplinewithUnits(Test_UnivariateSplinewithUnits):
    """Test LSQUnivariateSplinewithUnits."""

    @pytest.fixture
    def x(self, num):
        return np.linspace(0, 6, num=num) * u.deg

    @pytest.fixture
    def y(self, num, x):
        return (np.exp(-(x.value ** 2)) + 0.1) * u.m

    @pytest.fixture
    def extra_args(self):
        return dict(k=3, ext=0, check_finite=False)

    @pytest.fixture
    def ispline_cls(self):
        return interpolate.LSQUnivariateSplinewithUnits

    @pytest.fixture
    def t(self, ispline_cls, x, y, extra_args):
        spl = interpolate.InterpolatedUnivariateSplinewithUnits(
            x, y, w=None, bbox=[None] * 2, **extra_args
        )
        return spl.get_knots().value[1:-1]

    @pytest.fixture
    def spls(self, ispline_cls, x, y, w, t, bbox, extra_args):
        return dict(
            basic=ispline_cls(x, y, t, w=None, bbox=[None] * 2, **extra_args),
            weight=ispline_cls(x, y, t, w=w, bbox=[None] * 2, **extra_args),
            bbox=ispline_cls(x, y, t, w=None, bbox=bbox, **extra_args),
        )

    # ===============================================================
    # Method tests

    def test_fail_init(self, ispline_cls, x, y, t):
        """Test a failed initialization b/c wrong units."""
        with pytest.raises(u.UnitConversionError):
            bad_unit = x.unit / u.m
            ispline_cls(x, y, t, bbox=[0 * bad_unit, 180 * bad_unit])


##############################################################################
# END
