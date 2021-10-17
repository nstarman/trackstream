# -*- coding: utf-8 -*-

"""Testing :mod:`~trackstream.stream`."""

__all__ = [
    "TestStream",
]


##############################################################################
# IMPORTS

# THIRD PARTY
import astropy.coordinates as coord
import astropy.table as table
import astropy.units as u
import numpy as np
import pytest

# LOCAL
from trackstream.core import StreamTrack, TrackStream
from trackstream.example_data import get_example_pal5
from trackstream.stream import Stream

##############################################################################
# TESTS
##############################################################################


class TestStream:
    """Test :class:`trackstream.stream.Stream`."""

    @classmethod
    def setup_class(self):
        """Setup fixtures for testing."""
        self.stream_cls = Stream
        self.data = get_example_pal5()
        self.origin = self.data.meta["origin"]
        self.data_err = None  # TODO?
        self.frame = None

    @pytest.fixture
    def stream_cls(self):
        """Stream class."""
        return self.stream_cls

    @pytest.fixture(params=[None, True])
    def stream(self, stream_cls, request):
        """Stream instance."""
        frame = self.frame if request.param is True else request.param
        return stream_cls(self.data, self.origin, data_err=self.data_err, frame=frame)

    # ===============================================================

    def test_init_fail_numargs(self, stream_cls):
        """Test with wrong number of arguments. A reminder to include ``origin``."""
        with pytest.raises(TypeError, match="origin"):
            stream = stream_cls(self.data)

    def test_init(self, stream_cls):
        """Test initialization."""
        stream = stream_cls(self.data, self.origin, data_err=self.data_err, frame=self.frame)

        # origin
        assert isinstance(stream.origin, coord.SkyCoord)
        assert stream.origin == self.origin

        # frame
        assert stream._system_frame is self.frame

        # cache
        assert stream._cache == dict()

        # data
        # assert stream._original_data is None  # NOT! it's processed

        assert isinstance(stream.data, table.Table)

    # -------------------------------------------

    def test_system_frame(self, stream):
        """Test system-centric frame."""
        # if passed a frame at initialization
        if stream._system_frame is not None:  # if passed frame at init
            assert stream.system_frame is self._system_frame
        else:
            assert stream.system_frame is None  # == stream._cache.get("frame", None)
            # TODO! have test for fit streams, where it isn't None

    def test_frame(self, stream):
        """Test attribute ``frame``."""
        assert stream.frame is stream.system_frame

    def test_number_of_tails(self, stream):
        assert stream.number_of_tails == 2 if (stream.has_arm1 and stream.has_arm2) else 1

    def test_has_arm1(self, stream):
        flags = np.unique(stream.data["tail"])
        assert stream.has_arm1 is True if "arm_1" in flags else False

    def test_has_arm2(self, stream):
        flags = np.unique(stream.data["tail"])
        assert stream.has_arm2 is True if "arm_2" in flags else False

    def test_coords(self, stream):
        assert isinstance(stream.coords, coord.SkyCoord)

        frame = stream.system_frame if stream.system_frame is not None else stream.data_frame
        assert np.all(stream.coords == stream.data_coords.transform_to(frame))

    def test_coords_arm1(self, stream):
        got = stream.coords_arm1
        if stream.number_of_tails == 2:
            assert np.all(got == stream.coords[stream.data["tail"] == "arm_1"])
        else:
            assert got is stream.coords

    def test_coords_arm2(self, stream):
        got = stream.coords_arm2
        if stream.has_arm2:
            assert np.all(got == stream.coords[stream.data["tail"] == "arm_2"])
        else:
            assert got is None

    # -------------------------------------------

    # def test_original_data(self, stream):
    #     assert stream.original_data == stream._data_frame.realize_frame(stream._data)

    def test_data_coords(self, stream):
        assert np.all(stream.data_coords == stream.data["coord"])

    def test_data_frame(self, stream):
        assert np.all(stream.data_frame == stream.data_coords.frame.replicate_without_data())

    # -------------------------------------------

    @pytest.mark.skip("TODO!")
    def test_normalize_data(self, stream_cls):
        assert False

    # -------------------------------------------

    def test_track(self, stream):
        # Different test if not already fit a track
        if "track" not in stream._cache:
            with pytest.raises(ValueError, match="need to fit track"):
                stream.track

            stream.fit_track()

        # now guaranteed to have a working track
        assert isinstance(stream.track, StreamTrack)

        # TODO! more tests

    # ===============================================================
    # Test Usage

    @pytest.mark.skip("TODO!")
    def test_loading_pal5(self, stream_cls):
        data = get_example_pal5()
        stream = Stream(data, data.meta["origin"])
        assert False


# /class


##############################################################################
# END
