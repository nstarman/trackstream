# -*- coding: utf-8 -*-

"""Testing :mod:`~trackstream.utils.descriptors`."""


##############################################################################
# IMPORTS

# STDLIB
import weakref

# THIRD PARTY
import pytest

# LOCAL
from trackstream.utils.descriptors import InstanceDescriptor

##############################################################################
# TESTS
##############################################################################


class Test_InstanceDescriptor:
    """Test :class:`~trackstream.utils.descriptors.InstanceDescriptor`."""

    @pytest.fixture(scope="class")
    def descriptor_cls(self):
        return InstanceDescriptor

    @pytest.fixture(scope="class")
    def enclosing_attr(self):
        return "attr"

    @pytest.fixture(scope="class")
    def enclosing_cls(self, descriptor_cls):
        """Fixture of the enclosing class."""

        class Enclosing:
            attr = descriptor_cls["Enclosing"]()

            def __init__(self) -> None:
                self.x = 1

        return Enclosing

    @pytest.fixture(scope="class")
    def enclosing(self, enclosing_cls):
        """Fixture of the enclosing instance."""
        encl = enclosing_cls()
        return encl

    @pytest.fixture(scope="class")
    def thedescriptor(self, enclosing_cls, enclosing_attr):
        """Fixture of the instance descriptor."""
        return getattr(enclosing_cls, enclosing_attr)

    @pytest.fixture(scope="class")
    def descriptor(self, enclosing, enclosing_attr):
        """Fixture of the instance descriptor."""
        return getattr(enclosing, enclosing_attr)

    @pytest.fixture(scope="class")
    def isolated_descriptor(self, descriptor_cls):
        """Fixture of the instance descriptor."""
        return descriptor_cls()

    # ===============================================================
    # Method Tests

    def test_expected_attributes(self, descriptor):
        """Test the desciptor expects the right attributes."""
        annot = descriptor.__annotations__
        assert annot["_enclosing_attr"] is str
        assert type(None) in annot["_enclosing_ref"].__args__
        assert annot["_enclosing_ref"].__args__[0].__origin__ is weakref.ReferenceType

        # todo? test these are the ONLY attributes expected?

    def test_init(self, descriptor_cls):
        """Test the descriptor initialization."""
        descriptor = descriptor_cls()
        assert descriptor._enclosing_attr == ""
        assert descriptor._enclosing_ref is None

    def test_get_enclosing(self, descriptor, enclosing):
        """Test :meth:`~trackstream.utils.descriptors.InstanceDescriptor._get_enclosing`."""
        assert descriptor._get_enclosing() is enclosing

    def test_get_enclosing_None(self, isolated_descriptor):
        """Test :meth:`~trackstream.utils.descriptors.InstanceDescriptor._get_enclosing`."""
        assert isolated_descriptor._get_enclosing() is None

    def test_enclosing(self, descriptor, enclosing):
        """Test :attr:`~trackstream.utils.descriptors.InstanceDescriptor._enclosing`."""
        assert descriptor._get_enclosing() is enclosing

    def test_enclosing_None(self, isolated_descriptor):
        """Test :attr:`~trackstream.utils.descriptors.InstanceDescriptor._enclosing`."""
        with pytest.raises(ValueError, match="no reference"):
            isolated_descriptor._enclosing

    def test_set_name(self, descriptor_cls, enclosing_attr):
        """Test :attr:`~trackstream.utils.descriptors.InstanceDescriptor.__set_name__`."""
        descriptor = descriptor_cls()
        descriptor.__set_name__(None, enclosing_attr)

        assert descriptor._enclosing_attr == enclosing_attr

    def test_get(self, enclosing_cls, enclosing, enclosing_attr, thedescriptor, descriptor):
        """Test :attr:`~trackstream.utils.descriptors.InstanceDescriptor.__get__`."""
        # Getting the descriptor from the class
        descr = getattr(enclosing_cls, enclosing_attr)
        assert descr is thedescriptor
        assert descr is not descriptor

        # Getting the descriptor from the instance
        descr = getattr(enclosing, enclosing_attr)
        assert descr is descriptor
        assert descr is not thedescriptor
