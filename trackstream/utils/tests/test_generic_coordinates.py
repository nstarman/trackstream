# -*- coding: utf-8 -*-

"""Testing :mod:`~trackstream.utils.generic_coordinates`."""

__all__ = [
    "test__GENERIC_REGISTRY",
    "Test_GenericRepresentation",
]


##############################################################################
# IMPORTS

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import pytest

# LOCAL
from trackstream.utils import generic_coordinates as gcoord

##############################################################################
# TESTS
##############################################################################


def test__GENERIC_REGISTRY():
    """:obj:`~trackstream.utils.generic_coordinates._GENERIC_REGISTRY`.

    Just some easy tests.

    """
    assert isinstance(gcoord._GENERIC_REGISTRY, dict)

    for key, val in gcoord._GENERIC_REGISTRY.items():
        if not isinstance(key, str):
            assert issubclass(key, coord.BaseRepresentation)

        assert issubclass(
            val,
            (gcoord.GenericRepresentation, gcoord.GenericDifferential),
        )


#####################################################################


class Test_GenericRepresentation:
    """Test :class:`~{package}.{klass}`."""

    @pytest.fixture
    def rep_cls(self):
        return gcoord.GenericRepresentation

    # @pytest.fixture
    # def rep(self, rep_cls):
    #     return rep_cls(q1=1, q2=2, q3=3)
    @pytest.fixture
    def rep(self, rep_cls):
        class GenericRepresentationSubClass(rep_cls):
            attr_classes = dict(q1=u.Quantity, q2=u.Quantity, q3=u.Quantity)

            def from_cartesian(self):
                pass

            def to_cartesian(self):
                pass

        return GenericRepresentationSubClass(q1=1, q2=2, q3=3)

    # -------------------------------

    def test_attr_classes(self, rep_cls, rep):
        """Test attribute ``attr_classes``."""
        # works as class attribute
        assert tuple(rep_cls.attr_classes.keys()) == ("q1", "q2", "q3")

        # works as instance attribute
        assert tuple(rep.attr_classes.keys()) == ("q1", "q2", "q3")


# -------------------------------------------------------------------
# TODO tests for all the dynamically-defined subclasses


def test__make_generic_representation():
    """Test function ``_make_generic_representation``."""
    # ------------------
    # already generic

    got = gcoord._make_generic_representation(gcoord.GenericRepresentation)
    assert got is gcoord.GenericRepresentation  # pass thru unchanged

    # ------------------
    # need to make

    got = gcoord._make_generic_representation(coord.CartesianRepresentation)

    assert gcoord._GENERIC_REGISTRY  # not empty anymore
    assert got is gcoord.GenericCartesianRepresentation  # cached

    # ------------------
    # cached

    expected = got
    got = gcoord._make_generic_representation(coord.CartesianRepresentation)

    assert got is expected
    assert got is gcoord.GenericCartesianRepresentation  # cached


#####################################################################


class TestGenericDifferential:
    """Test :class:`~{package}.{klass}`."""

    @pytest.fixture
    def rep_cls(self):
        return gcoord.GenericRepresentation

    @pytest.fixture
    def dif_cls(self):
        return gcoord.GenericDifferential

    # @pytest.fixture
    # def dif(self, dif_cls):
    #     return dif_cls(d_q1=1, d_q2=2, d_q3=3)
    @pytest.fixture
    def dif(self, dif_cls, rep_cls):
        class GenericDifferentialSubClass(dif_cls):
            base_representation = rep_cls

            def from_cartesian(self):
                pass

            def to_cartesian(self):
                pass

        return GenericDifferentialSubClass(d_q1=1, d_q2=2, d_q3=3)

    # -------------------------------

    def test_base_representation(self, dif_cls, dif):
        """Test attribute ``attr_classes``."""
        # works as class attribute
        assert dif_cls.base_representation == gcoord.GenericRepresentation

        # works as instance attribute
        assert dif.base_representation == gcoord.GenericRepresentation


# -------------------------------------------------------------------
# TODO tests for all the dynamically-defined subclasses


def test__make_generic_differential():
    """Test function ``_make_generic_differential``."""
    # ------------------
    # already generic

    got = gcoord._make_generic_differential(gcoord.GenericDifferential)
    assert got is gcoord.GenericDifferential  # pass thru unchangeds

    # ------------------
    # n too small

    with pytest.raises(ValueError):
        gcoord._make_generic_differential(coord.SphericalDifferential, n=0)

    # ------------------
    # need to make, n=1

    got = gcoord._make_generic_differential(coord.SphericalDifferential, n=1)
    assert got is gcoord.GenericSphericalDifferential  # cached

    # ------------------
    # need to make, n=2

    got = gcoord._make_generic_differential(coord.SphericalDifferential, n=2)
    assert got is gcoord.GenericSpherical2ndDifferential  # cached

    # ------------------
    # cached

    got = gcoord._make_generic_differential(coord.SphericalDifferential, n=1)
    assert got is gcoord.GenericSphericalDifferential  # cached


# -------------------------------------------------------------------


def test__make_generic_differential_for_representation():
    """Test function ``_make_generic_differential_for_representation``."""
    # ------------------
    # n=1, and not in generics' registry

    got = gcoord._make_generic_differential_for_representation(
        coord.PhysicsSphericalRepresentation,
        n=1,
    )
    assert got is gcoord.GenericPhysicsSphericalDifferential

    # ------------------
    # do again, getting from registry

    got = gcoord._make_generic_differential_for_representation(
        coord.PhysicsSphericalRepresentation,
        n=1,
    )
    assert got is gcoord.GenericPhysicsSphericalDifferential

    # ------------------
    # n=2, and not in generics' registry

    got = gcoord._make_generic_differential_for_representation(
        coord.PhysicsSphericalRepresentation,
        n=2,
    )
    assert got is gcoord.GenericPhysicsSpherical2ndDifferential

    # ------------------
    # do again, getting from registry

    got = gcoord._make_generic_differential_for_representation(
        coord.PhysicsSphericalRepresentation,
        n=2,
    )
    assert got is gcoord.GenericPhysicsSpherical2ndDifferential


##############################################################################
# END
