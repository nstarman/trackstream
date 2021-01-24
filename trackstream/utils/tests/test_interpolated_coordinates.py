# -*- coding: utf-8 -*-

"""Testing :mod:`~trackstream.utils.interpolated_coordinates`."""

__all__ = []


##############################################################################
# IMPORTS

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import pytest
from astropy.utils.decorators import format_doc

# PROJECT-SPECIFIC
from trackstream.tests.helper import BaseClassDependentTests
from trackstream.utils import interpolated_coordinates as icoord

##############################################################################
# TESTS
##############################################################################


@format_doc(
    None,
    package="trackstream.utils.interpolated_coordinates",
    klass="InterpolatedRepresentationOrDifferential",
)
class Test_InterpolatedRepresentationOrDifferential(
    BaseClassDependentTests,
    klass=icoord.InterpolatedRepresentationOrDifferential,
):
    """Test :class:`~{package}.{klass}`."""

    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        num = 40
        cls.affine = np.linspace(0, 10, num=num) * u.Myr

    # /def

    # -------------------------------

    def test_init(self):
        with pytest.raises(TypeError) as e:
            self.klass()

        assert "Cannot instantiate" in str(e.value)

    # /def


# /class


# -------------------------------------------------------------------


@pytest.mark.skip("TODO")
@format_doc(
    None,
    package="trackstream.utils.interpolated_coordinates",
    klass="InterpolatedRepresentationOrDifferential",
)
class InterpolatedRepresentationOrDifferentialTests(
    Test_InterpolatedRepresentationOrDifferential,
    klass=icoord.InterpolatedRepresentationOrDifferential,
):
    """Tests for :class:`~{package}.{klass}` subclasses."""

    # ---------------------------------------------------------------
    # Public attributes / properties

    def _test_data(self, obj):
        assert isinstance(obj, self.klass)
        assert isinstance(obj.data, coord.BaseRepresentationOrDifferential)

    def test_data(self):
        for inst in self.insts:

            self._test_data(inst)

            if hasattr(inst.data, "differentials"):
                for differential in inst.data.differentials.values():
                    self._test_data(differential)

    # /def

    def test_affine(self):
        for inst in self.insts:

            assert u.allclose(inst.affine, self.affine)

    # /def

    def test_derivative_type(self):
        for inst in self.insts:

            assert False, inst.derivative_type

    # /def

    # ---------------------------------------------------------------
    # Public methods

    def test_clear_derivatives(self):

        for inst in self.insts:

            inst.derivative(n=1)  # ensure has derivative

            assert inst.derivatives  # not empty

            self.clear_derivatives()

            assert not inst.derivatives  # empty

    # /def

    def test_derivative(self):

        assert False

    def test_from_cartesian(self):

        assert False

    def test_to_cartesian(self):

        assert False

    def test_copy(self):

        assert False

    # ---------------------------------------------------------------
    # Property-based tests

    def test_duck_class(self):

        for inst in self.insts:

            assert isinstance(inst, self.base_klass)

            # just to make sure!
            assert isinstance(inst.__class__, self.base_klass)

    # /def

    def test_get_from_underlying_data(self):

        for inst in self.insts:

            # make sure it's not defined on the class
            assert "info" not in inst.__dict__

            # test from underlying class
            assert inst.info == inst.data.info

    # /def

    def test_math_methods(self):

        assert False

    # ---------------------------------------------------------------
    # Actions / pipe-line tests

    def test_changing_derivative_type(self):

        for inst in self.insts:

            inst.derivative(n=1)

            assert inst.derivatives  # not empty

            # change derivative tye
            inst.derivative_type = inst.__class__.compatible_differentials[0]

            # test derivative cache is cleared
            assert not inst.derivatives  # empty

    # /def

    def test_caching_derivatives(self):

        for inst in self.insts:

            inst.derivative(n=1)  # ensure has derivative
            inst.derivative(n=2)  # ensure has derivative

            assert "lambda 1" in inst.derivatives
            assert "lambda 2" in inst.derivatives

    # /def

    # def test_slicing_reinterpolates(self):

    #     for inst in self.insts:

    def test_evaluate_interpolation(self):
        """The call method"""

        assert False

    def test_higher_derivatives(self):

        assert False


# /class


#####################################################################


@pytest.mark.skip("TODO")
@format_doc(
    None,
    package="trackstream.utils.interpolated_coordinates",
    klass="InterpolatedRepresentation",
)
class Test_InterpolatedRepresentation(
    InterpolatedRepresentationOrDifferentialTests,
    klass=icoord.InterpolatedRepresentation,
):
    """Test :class:`~{package}.{klass}`."""

    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        # num = 40

        # cls.x = np.linspace(0, 180, num=num) * u.deg
        # cls.y = np.linspace(0, 10, num=num) * u.m
        # cls.w = np.random.rand(num)
        # cls.bbox = [0 * u.deg, 180 * u.deg]
        # cls.extra_args = extra_args = dict(
        #     s=None, k=3, ext=0, check_finite=False
        # )

        # cls.spls = dict(
        #     basic=cls.klass(
        #         cls.x, cls.y, w=None, bbox=[None] * 2, **extra_args
        #     ),
        #     weight=cls.klass(
        #         cls.x, cls.y, w=cls.w, bbox=[None] * 2, **extra_args
        #     ),
        #     bbox=cls.klass(cls.x, cls.y, w=None, bbox=cls.bbox, **extra_args),
        # )

    # /def

    # -------------------------------

    def test_init(self):

        # Test normal
        assert False

        # special case of CartesianRepresenation
        assert False

    # /def


# /class


# -------------------------------------------------------------------


##############################################################################
# END
