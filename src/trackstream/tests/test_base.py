# -*- coding: utf-8 -*-

"""Initiation Tests for :mod:`trackstream.base`."""

# THIRD PARTY
import pytest
from astropy.coordinates import CartesianDifferential, CartesianRepresentation
from astropy.coordinates import SphericalDifferential, SphericalRepresentation
from astropy.coordinates import UnitSphericalDifferential

# LOCAL
from trackstream.base import CommonBase

##############################################################################
# TESTS
##############################################################################


class Test_CommonBase:
    """Test :class:`trackstream.base.CommonBase`."""

    @pytest.fixture(scope="class")
    def kls(self):
        return CommonBase

    @pytest.fixture(scope="class")
    def inst(self, kls, frame):
        return kls(frame=frame, representation_type=None, differential_type=None)

    # ===============================================================
    # Method Tests

    @pytest.mark.parametrize(
        ("rt", "dt"),
        [
            (None, None),
            (CartesianRepresentation, CartesianDifferential),
            (SphericalRepresentation, SphericalDifferential),
            (SphericalRepresentation, UnitSphericalDifferential),
        ],
    )
    def test_init(self, kls, frame, rt, dt):
        """Test initialization."""
        inst = kls(frame=frame, representation_type=rt, differential_type=dt)

        assert isinstance(inst.frame, type(frame))
        if rt is not None:
            assert inst.representation_type is rt
        if dt is not None:
            assert inst.differential_type is dt

    def test_frame(self, inst, frame):
        """Test :meth:`trackstream.base.CommonBase.frame`."""
        assert isinstance(inst.frame, type(frame))

    def test_representation_type(self, inst, rep_type):
        """Test :meth:`trackstream.base.CommonBase.representation_type`."""
        assert inst.representation_type is rep_type

    def test_differential(self, inst, dif_type):
        """Test :meth:`trackstream.base.CommonBase.differential_type`."""
        assert inst.differential_type is dif_type

    def test_rep_attrs(self, inst, rep_type):
        expected = tuple(getattr(rep_type, "attr_classes", {}).keys())
        assert inst._rep_attrs == expected

    def test_dif_attrs(self, inst, dif_type):
        expected = tuple(getattr(dif_type, "attr_classes", {}).keys())
        assert inst._dif_attrs == expected
