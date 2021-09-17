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


class Test_RepresentationOrDifferentialType(
    TypeVarTests,
    klass=_type_hints.RepresentationOrDifferentialType,
):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        cls.bound = coord.BaseRepresentationOrDifferential

    # /def


# /class

# -------------------------------------------------------------------


class Test_RepresentationType(
    TypeVarTests,
    klass=_type_hints.RepresentationType,
):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        cls.bound = coord.BaseRepresentation

    # /def


# /class

# -------------------------------------------------------------------


class Test_DifferentialType(TypeVarTests, klass=_type_hints.DifferentialType):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        cls.bound = coord.BaseDifferential

    # /def


# /class

# -------------------------------------------------------------------


class Test_FrameType(TypeVarTests, klass=_type_hints.FrameType):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        cls.bound = coord.BaseCoordinateFrame

    # /def

    def test_name(self):
        """Test that name is {bound}."""
        name: str = self.klass.__name__
        if name.startswith("~"):
            name = name[1:]

        assert name == "CoordinateFrame"

    # /def


# /class

# -------------------------------------------------------------------


class Test_SkyCoordType(TypeVarTests, klass=_type_hints.SkyCoordType):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        cls.bound = coord.SkyCoord

    # /def


# /class

# -------------------------------------------------------------------


@pytest.mark.skip("TODO")
class Test_CoordinateType:
    """Test CoordinateType."""


# /class

# -------------------------------------------------------------------


@pytest.mark.skip("TODO")
class Test_GenericPositionType:
    """Test GenericPositionType."""


# /class

# -------------------------------------------------------------------


@pytest.mark.skip("TODO")
class Test_FrameLikeType:
    """Test FrameLikeType."""


# /class

# -------------------------------------------------------------------


class Test_TableType(TypeVarTests, klass=_type_hints.TableType):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        cls.bound = table.Table

    # /def


# /class

# -------------------------------------------------------------------


class Test_QTableType(TypeVarTests, klass=_type_hints.QTableType):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        cls.bound = table.QTable

    # /def


# /class

# -------------------------------------------------------------------


@pytest.mark.skip("TODO")
class Test_UnitType:
    """Test UnitType."""


# /class


# -------------------------------------------------------------------


@pytest.mark.skip("TODO")
class Test_UnitLkeType:
    """Test UnitLkeType."""


# /class


# -------------------------------------------------------------------


class Test_QuantityType(TypeVarTests, klass=_type_hints.QuantityType):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        cls.bound = u.Quantity

    # /def


# /class

# -------------------------------------------------------------------


@pytest.mark.skip("TODO")
class Test_QuantityLkeType:
    """Test QuantityLkeType."""


# /class

# -------------------------------------------------------------------


##############################################################################
# END
