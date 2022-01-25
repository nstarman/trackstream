# -*- coding: utf-8 -*-

"""Testing :mod:`~trackstream.utils.generic_coordinates`."""

__all__ = [
    "test_GENERIC_REGISTRY",
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


def test_GENERIC_REGISTRY():
    """Test :obj:`~trackstream.utils.generic_coordinates._GENERIC_REGISTRY`."""
    # Check type
    assert isinstance(gcoord._GENERIC_REGISTRY, dict)

    # Check entries
    for key, val in gcoord._GENERIC_REGISTRY.items():
        if not isinstance(key, str):
            assert issubclass(key, coord.BaseRepresentationOrDifferential)

        assert issubclass(
            val,
            (gcoord.GenericRepresentation, gcoord.GenericDifferential),
        )


#####################################################################


class Test_GenericRepresentation:
    """Test `trackstream.utils.generic_coordinates.GenericRepresentation`."""

    @pytest.fixture(scope="class")
    def rep_cls(self):
        return gcoord.GenericRepresentation

    # ===============================================================

    def test_attr_classes(self, rep_cls):
        """Test attribute ``attr_classes``."""
        # works as class attribute
        assert tuple(rep_cls.attr_classes.keys()) == ("q1", "q2", "q3")

    def test_init(self, rep_cls):
        """It's abstract, so can't init."""
        with pytest.raises(TypeError, match="Can't instantiate abstract"):
            rep_cls(q1=1, q2=2, q3=3)

    def test_make_generic_cls(self, rep_cls):
        """Test function ``_make_generic_cls``."""
        # ------------------
        # Already generic

        got = rep_cls._make_generic_cls(gcoord.GenericRepresentation)
        assert got is gcoord.GenericRepresentation  # pass thru unchanged

        # ------------------
        # Need to make (or cached)

        got = rep_cls._make_generic_cls(coord.CartesianRepresentation)

        assert gcoord._GENERIC_REGISTRY  # not empty anymore
        assert got is gcoord.GenericCartesianRepresentation  # cached

        # ------------------
        # Definitely cached

        expected = got
        got = rep_cls._make_generic_cls(coord.CartesianRepresentation)

        assert got is expected
        assert got is gcoord.GenericCartesianRepresentation  # cached


class Test_GenericRepresentationSubclass(Test_GenericRepresentation):
    @pytest.fixture(scope="class")
    def rep_cls(self):
        class GenericRepresentationSubClass(gcoord.GenericRepresentation):
            attr_classes = dict(q1=u.Quantity, q2=u.Quantity, q3=u.Quantity)

            def from_cartesian(self):
                pass

            def to_cartesian(self):
                pass

        return GenericRepresentationSubClass

    @pytest.fixture(scope="class")
    def rep(self, rep_cls):
        return rep_cls(q1=1, q2=2, q3=3)

    # ===============================================================

    def test_attr_classes(self, rep_cls, rep):
        """Test attribute ``attr_classes``."""
        super().test_attr_classes(rep_cls)

        # works as instance attribute
        assert tuple(rep.attr_classes.keys()) == ("q1", "q2", "q3")

    @pytest.mark.parametrize(
        "q1, q2, q3",
        [
            (1, 2, 3),  # no units
            (1 * u.km, 2, 3),  # mixed units
            (1, 2, 3) * u.deg,  # all units
        ],
    )
    def test_init(self, rep_cls, q1, q2, q3):
        rep = rep_cls(q1=q1, q2=q2, q3=q3)
        assert (rep.q1, rep.q2, rep.q3) == (q1, q2, q3)


class Test_GenericCartesianRepresentation(Test_GenericRepresentation):
    @pytest.fixture(scope="class")
    def rep_cls(self):
        return gcoord.GenericRepresentation._make_generic_cls(coord.CartesianRepresentation)

    @pytest.fixture(scope="class")
    def rep(self, rep_cls):
        return rep_cls(x=1, y=2, z=3)


#####################################################################


class Test_GenericDifferential:
    """Test `trackstream.utils.generic_coordinates.GenericDifferential`."""

    @pytest.fixture(scope="class")
    def rep_cls(self):
        return gcoord.GenericRepresentation

    @pytest.fixture(scope="class")
    def dif_cls(self):
        return gcoord.GenericDifferential

    @pytest.fixture(scope="class")
    def dif(self, dif_cls):
        return dif_cls(d_q1=1, d_q2=2, d_q3=3)

    # ===============================================================

    def test_attr_classes(self, dif_cls):
        """Test attribute ``attr_classes``."""
        # works as class attribute
        assert tuple(dif_cls.attr_classes.keys()) == ("d_q1", "d_q2", "d_q3")

    def test_base_representation(self, dif_cls, dif):
        """Test attribute ``attr_classes``."""
        # works as class attribute
        assert dif_cls.base_representation == gcoord.GenericRepresentation

        # works as instance attribute
        assert dif.base_representation == gcoord.GenericRepresentation

    @pytest.mark.parametrize(
        "q1, q2, q3",
        [
            (1, 2, 3),  # no units
            (1 * u.km, 2, 3),  # mixed units
            (1, 2, 3) * u.deg,  # all units
        ],
    )
    def test_init(self, rep_cls, q1, q2, q3):
        rep = rep_cls(q1=q1, q2=q2, q3=q3)
        assert (rep.q1, rep.q2, rep.q3) == (q1, q2, q3)

    # ---------------------------------------------------------------

    def test_make_generic_cls(self, dif_cls):
        """Test function ``_make_generic_cls``."""
        # ------------------
        # already generic

        got = dif_cls._make_generic_cls(gcoord.GenericDifferential)
        assert got is gcoord.GenericDifferential  # pass thru unchangeds

        # ------------------
        # n too small

        with pytest.raises(ValueError):
            dif_cls._make_generic_cls(coord.SphericalDifferential, n=0)

        # ------------------
        # need to make, n=1

        got = dif_cls._make_generic_cls(coord.SphericalDifferential, n=1)
        assert got is gcoord.GenericSphericalDifferential  # cached

        # ------------------
        # need to make, n=2

        got = dif_cls._make_generic_cls(coord.SphericalDifferential, n=2)
        assert got is gcoord.GenericSpherical2ndDifferential  # cached

        # ------------------
        # cached

        got = dif_cls._make_generic_cls(coord.SphericalDifferential, n=1)
        assert got is gcoord.GenericSphericalDifferential  # cached

    def test_make_generic_cls_for_representation(self, dif_cls):
        """Test function ``_make_generic_cls_for_representation``."""
        # ------------------
        # n=1, and not in generics' registry

        got = dif_cls._make_generic_cls_for_representation(
            coord.PhysicsSphericalRepresentation,
            n=1,
        )
        assert got is gcoord.GenericPhysicsSphericalDifferential

        # ------------------
        # do again, getting from registry

        got = dif_cls._make_generic_cls_for_representation(
            coord.PhysicsSphericalRepresentation,
            n=1,
        )
        assert got is gcoord.GenericPhysicsSphericalDifferential

        # ------------------
        # n=2, and not in generics' registry

        got = dif_cls._make_generic_cls_for_representation(
            coord.PhysicsSphericalRepresentation,
            n=2,
        )
        assert got is gcoord.GenericPhysicsSpherical2ndDifferential

        # ------------------
        # do again, getting from registry

        got = dif_cls._make_generic_cls_for_representation(
            coord.PhysicsSphericalRepresentation,
            n=2,
        )
        assert got is gcoord.GenericPhysicsSpherical2ndDifferential


class Test_GenericDifferentialSubClass:
    @pytest.fixture(scope="class")
    def rep_cls(self):
        return gcoord.GenericRepresentation

    @pytest.fixture(scope="class")
    def dif_cls(self, rep_cls):
        class GenericDifferentialSubClass(gcoord.GenericDifferential):
            base_representation = rep_cls

            def from_cartesian(self):
                pass

            def to_cartesian(self):
                pass

        return GenericDifferentialSubClass

    @pytest.fixture(scope="class")
    def dif(self, dif_cls):
        return dif_cls(d_q1=1, d_q2=2, d_q3=3)


class TestGenericCartesianDifferential:
    @pytest.fixture(scope="class")
    def rep_cls(self):
        return coord.CartesianDifferential

    @pytest.fixture(scope="class")
    def dif_cls(self, rep_cls):
        return gcoord.GenericDifferential._make_generic_cls(rep_cls)

    @pytest.fixture(scope="class")
    def dif(self, dif_cls):
        return dif_cls(d_x=1, d_y=2, d_z=3)


class TestGenericCylindrical3rdDifferential:
    @pytest.fixture(scope="class")
    def rep_cls(self):
        return coord.CylindricalDifferential

    @pytest.fixture(scope="class")
    def dif_cls(self, rep_cls):
        return gcoord.GenericDifferential._make_generic_cls(rep_cls)

    @pytest.fixture(scope="class")
    def dif(self, dif_cls):
        return dif_cls(d_rho=1, d_phi=2, d_z=3)


##############################################################################
# END
