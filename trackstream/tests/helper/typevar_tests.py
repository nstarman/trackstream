# -*- coding: utf-8 -*-

"""`~typing.TypeVar` Tests."""

__all__ = [
    "TypeVarTests",
]


##############################################################################
# IMPORTS

# BUILT-IN
import abc
import typing as T

# PROJECT-SPECIFIC
from .baseclassdependent import BaseClassDependentTests

##############################################################################
# CODE
##############################################################################


class TypeVarTests(BaseClassDependentTests, klass=T.TypeVar):
    """Type Testing Framework."""

    @classmethod
    @abc.abstractmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        pass

    # /def

    # -------------------------------

    def test_isTypeVar(self):
        """Test that this is a TypeVar."""
        assert isinstance(self.klass, T.TypeVar)

    # /def

    def test_bound(self):
        """Test TypeVar is correctly bound."""
        assert self.klass.__bound__ is self.bound

    # /def

    def test_name(self):
        """Test that name is [bound]."""
        name: str = self.klass.__name__
        if name.startswith("~"):
            name = name[1:]

        boundname: str = self.bound.__name__

        assert name == f"{boundname}", f"{name} != {boundname}"

    # /def


# /class


# -------------------------------------------------------------------

##############################################################################
# END
