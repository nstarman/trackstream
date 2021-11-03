# -*- coding: utf-8 -*-

"""Testing :mod:`~discO._type_hints`."""

__all__ = [
    # Astropy types
    # coordinates
    "Test_RepresentationOrDifferentialType",
    "Test_RepresentationType",
    "Test_DifferentialType",
    "Test_FrameType",
    "Test_SkyCoordType",
    "Test_CoordinateType",
    "Test_GenericPositionType",
    "Test_FrameLikeType",
    # tables
    "Test_TableType",
    "Test_QTableType",
    # units
    "Test_UnitType",
    "Test_UnitLkeType",
    "Test_QuantityType",
    "Test_QuantityLkeType",
]


##############################################################################
# IMPORTS

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import pytest
from astropy import table

# LOCAL
from trackstream import _type_hints
from trackstream.tests.helper import TypeVarTests

##############################################################################
# TESTS
##############################################################################


class Test_RepresentationOrDifferentialType(TypeVarTests):
    @classmethod
    def setup_class(self):
        """Setup fixtures for testing."""
        self.bound = coord.BaseRepresentationOrDifferential


class Test_RepresentationType(TypeVarTests):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        cls.bound = coord.BaseRepresentation


class Test_DifferentialType(TypeVarTests):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        cls.bound = coord.BaseDifferential


class Test_FrameType(TypeVarTests):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        cls.bound = coord.BaseCoordinateFrame

    def test_name(self):
        """Test that name is {bound}."""
        name: str = self.klass.__name__
        if name.startswith("~"):
            name = name[1:]

        assert name == "CoordinateFrame"


class Test_SkyCoordType(TypeVarTests):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        cls.bound = coord.SkyCoord


@pytest.mark.skip("TODO")
class Test_CoordinateType:
    """Test CoordinateType."""


@pytest.mark.skip("TODO")
class Test_GenericPositionType:
    """Test GenericPositionType."""


@pytest.mark.skip("TODO")
class Test_FrameLikeType:
    """Test FrameLikeType."""


class Test_TableType(TypeVarTests):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        cls.bound = table.Table


class Test_QTableType(TypeVarTests):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        cls.bound = table.QTable


@pytest.mark.skip("TODO")
class Test_UnitType:
    """Test UnitType."""


@pytest.mark.skip("TODO")
class Test_UnitLkeType:
    """Test UnitLkeType."""


class Test_QuantityType(TypeVarTests):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        cls.bound = u.Quantity


@pytest.mark.skip("TODO")
class Test_QuantityLkeType:
    """Test QuantityLkeType."""


##############################################################################
# END
