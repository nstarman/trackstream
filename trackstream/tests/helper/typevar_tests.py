# -*- coding: utf-8 -*-

"""`~typing.TypeVar` Tests."""

__all__ = ["TypeVarTests"]


##############################################################################
# IMPORTS

# STDLIB
import abc
import typing as T

# THIRD PARTY
import pytest

##############################################################################
# CODE
##############################################################################


class TypeVarTests:
    """Type Testing Framework."""

    @classmethod
    @abc.abstractmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        pass

    @pytest.fixture
    def type_cls(self):
        return T.TypeVar

    # -------------------------------

    def test_isTypeVar(self, type_cls):
        """Test that this is a TypeVar."""
        assert isinstance(type_cls, T.TypeVar)

    def test_bound(self, type_cls):
        """Test TypeVar is correctly bound."""
        assert type_cls.__bound__ is self.bound

    def test_name(self, type_cls):
        """Test that name is [bound]."""
        name: str = type_cls.__name__
        if name.startswith("~"):
            name = name[1:]

        boundname: str = self.bound.__name__

        assert name == f"{boundname}", f"{name} != {boundname}"


##############################################################################
# END
