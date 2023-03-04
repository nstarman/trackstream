"""Testing :mod:`~trackstream.stream.base`."""

from abc import ABCMeta, abstractmethod
from types import NoneType

from astropy.coordinates import BaseCoordinateFrame, SkyCoord
from astropy.table import QTable
from attr import define
import numpy as np
import pytest

from trackstream.stream.base import StreamBase

##############################################################################
# TESTS
##############################################################################


class StreamBaseTest(metaclass=ABCMeta):
    """Test :class:`trackstream.stream.Stream`.

    .. todo::

        Tests for ``_data_max_lines``.

    """

    @pytest.fixture(scope="class")
    def stream_cls(self):
        """Stream class."""
        return StreamBase

    @pytest.fixture(scope="class")
    @abstractmethod
    def data_table(self):
        return None  # get_example_pal5()

    @pytest.fixture(scope="class")
    def data_error_table(self):
        return None  # TODO!

    @pytest.fixture(scope="class")
    def origin(self, data_table):
        return None  # TODO!

    @pytest.fixture(scope="class")
    def frame(self):
        return None  # TODO! more options

    @pytest.fixture(scope="class")
    @abstractmethod
    def stream(self, stream_cls):
        pass

    # def stream(self, stream_cls, data_table, data_error_table, origin, frame):
    #     """Stream instance."""

    # ===============================================================

    @abstractmethod
    def test_data(self, stream):
        """Test property ``data``."""
        assert isinstance(stream.data, QTable)

    @abstractmethod
    def test_data_frame(self, stream):
        """Test property ``data_frame``."""
        assert isinstance(stream.data_frame, BaseCoordinateFrame)

    @abstractmethod
    def test_coords(self, stream):
        """Test property ``coords``."""
        assert isinstance(stream.coords, SkyCoord)

    @abstractmethod
    def test_coords_ord(self, stream):
        """Test property ``coords_ord``."""
        assert isinstance(stream.coords_ord, SkyCoord)

    @abstractmethod
    def test_frame(self, stream):
        """Test property ``frame``."""
        assert isinstance(stream.frame, (BaseCoordinateFrame, NoneType))

    @abstractmethod
    def test_name(self, stream):
        """Test property ``name``."""
        assert isinstance(stream.name, (str, NoneType))

    @abstractmethod
    def test_origin(self, stream):
        """Test property ``origin``."""
        assert isinstance(stream.origin, SkyCoord)

    @abstractmethod
    def test_has_distances(self, stream):
        """Test property ``has_distances``."""
        hasds = stream.coords.spherical.distance.unit.physical_type == "length"
        assert stream.has_distances is hasds

    @abstractmethod
    def test_full_name(self, stream):
        """Test property ``full_name``."""
        assert isinstance(stream.full_name, (str, NoneType))
        assert stream.full_name is stream.name

    # -------------------------------------------
    # Magic methods

    @pytest.mark.skip("TODO!")
    def test_repr(self, stream):
        """Test ``repr(stream)``."""

    def test_len(self, stream):
        """Test ``len(stream)``"""
        assert len(stream) == len(stream.data)


#####################################################################


class Test_StreamBase(StreamBaseTest):
    @pytest.fixture(scope="class")
    def stream_cls(self):
        """Stream class."""

        @define
        class StreamEx(StreamBase):
            data: QTable
            _data_max_lines = 10

            def __init__(self, data) -> None:
                object.__setattr__(self, "data", data)

            @property
            def data(self):
                return super().data

            @property
            def data_frame(self):
                return super().data_frame

            @property
            def coords(self):
                return super().coords

            @property
            def coords_ord(self):
                return super().coords_ord

            @property
            def system_frame(self):
                return super().system_frame

            @property
            def name(self):
                return super().name

            @property
            def origin(self):
                return super().origin

        return StreamEx

    @pytest.fixture(scope="class")
    def data_table(self):
        return np.array([1, 2, 3])

    @pytest.fixture(scope="class")
    def stream(self, stream_cls, data_table):
        return stream_cls(data_table)

    # ===============================================================

    @pytest.mark.skip("TODO!")
    def test_data(self, stream):
        """Test property ``data``."""
        raise AssertionError

    @pytest.mark.skip("TODO!")
    def test_data_frame(self, stream):
        """Test property ``data_frame``."""
        raise AssertionError

    @pytest.mark.skip("TODO!")
    def test_coords(self, stream):
        """Test property ``coords``."""
        raise AssertionError

    @pytest.mark.skip("TODO!")
    def test_coords_ord(self, stream):
        """Test property ``coords_ord``."""
        raise AssertionError

    @pytest.mark.skip("TODO!")
    def test_frame(self, stream):
        """Test property ``frame``."""
        raise AssertionError

    @pytest.mark.skip("TODO!")
    def test_name(self, stream):
        """Test property ``name``."""
        raise AssertionError

    @pytest.mark.skip("TODO!")
    def test_origin(self, stream):
        """Test property ``origin``."""
        raise AssertionError

    @pytest.mark.skip("TODO!")
    def test_has_distances(self, stream):
        """Test property ``has_distances``."""
        raise AssertionError

    @pytest.mark.skip("TODO!")
    def test_full_name(self, stream):
        """Test property ``full_name``."""
        raise AssertionError

    def test_len(self, stream, data_table):
        """Test ``len(stream)``"""
        assert len(stream) == len(data_table)
