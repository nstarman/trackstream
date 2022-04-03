# -*- coding: utf-8 -*-

"""Testing :class:`trackstream.core.TrackStream`."""

__all__ = ["Test_TrackStream"]


##############################################################################
# IMPORTS

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import pytest
from astropy.tests.helper import assert_quantity_allclose

# LOCAL
from trackstream.core import StreamTrack, TrackStream
from trackstream.example_data import get_example_pal5
from trackstream.som import SelfOrganizingMap1D
from trackstream.stream import Stream

##############################################################################
# TESTS
##############################################################################


class Test_TrackStream:
    """Test :class:`~trackstream.core.TrackStream`."""

    @classmethod
    def setup_class(self):
        """Setup fixtures for testing."""
        self.stream_cls = Stream
        self.data = get_example_pal5()
        self.origin = self.data.meta["origin"]
        self.data_err = None  # TODO?
        self.frame = None

    @pytest.fixture(params=[(None, None), (True, None), (None, True), (True, True)])
    def tracker(self, request):
        arm1SOM, arm2SOM = request.param
        if arm1SOM is True:
            # TODO! have a test for a pre-made SOM
            arm1SOM = None
        if arm2SOM is True:
            # TODO! have a test for a pre-made SOM
            arm2SOM = None

        return TrackStream(arm1SOM=arm1SOM, arm2SOM=arm2SOM)

    @pytest.fixture
    def tracker_cls(self):
        return TrackStream

    @pytest.fixture
    def stream_cls(self):
        """Stream class."""
        return self.stream_cls

    @pytest.fixture
    def stream(self, stream_cls):
        """Stream instance."""
        frame = self.frame
        return stream_cls(self.data, self.origin, data_err=self.data_err, frame=frame)

    # ===============================================================
    # Method tests

    def test_init(self, tracker):
        """Test instantiation."""
        assert isinstance(tracker._cache, dict)
        assert isinstance(tracker._arm1_SOM, (type(None), SelfOrganizingMap1D))
        assert isinstance(tracker._arm2_SOM, (type(None), SelfOrganizingMap1D))

    # -------------------------------

    def test_fit(self, tracker, stream):
        """Test method ``fit``."""
        track = tracker.fit(stream)

        assert isinstance(track, StreamTrack)
        # TODO! a lot more tests

    def test_predict(self, tracker):
        """Test method ``predict``."""
        arclength = np.linspace(0, 1)
        with pytest.raises(AttributeError):  # can't call what don't have
            tracker.predict(arclength)

    def test_fit_predict(self, tracker, stream):
        """Test method ``fit_predict``."""
        arclength = np.linspace(0, 1)
        with pytest.raises(AttributeError):  # can't call what don't have
            tracker.fit_predict(stream, arclength)


# /class


# -------------------------------------------------------------------


class Test_StreamTrack:
    """Test :class:`~trackstream.core.StreamTrack`."""

    #     @classmethod
    #     def setup_class(self):
    #         """Setup fixtures for testing."""
    #         # TODO! move to
    #         num = 40
    #         self.arclength = np.linspace(0, 10, num=num) * u.deg
    #
    #         lon = np.linspace(0, 25, num=num) * u.deg
    #         lat = np.linspace(-10, 10, num=num) * u.deg
    #         distance = np.linspace(8, 15, num=num) * u.kpc
    #
    #         self.data = coord.ICRS(
    #             coord.SphericalRepresentation(lon=lon, lat=lat, distance=distance),
    #         )
    #         self.interps = dict(
    #             lon=IUSU(self.arclength, lon),
    #             lat=IUSU(self.arclength, lat),
    #             distance=IUSU(self.arclength, distance),
    #         )
    #
    #         # origin
    #         i = num // 2
    #         self.origin = coord.ICRS(ra=lon[i], dec=lat[i])

    @pytest.fixture
    def origin(self, scrd, num):
        i = num // 2
        o = scrd[i]
        return o

    @pytest.fixture
    def track_cls(self):
        return StreamTrack

    @pytest.fixture
    def track(self, track_cls, path, scrd, origin, frame):
        """path and stream_data don't have to match up this nicely."""
        track = track_cls(path, stream_data=scrd, origin=origin, frame=frame)
        return track

    # ===============================================================
    # Method tests

    #     def test_init(self, track_cls, frame):
    #         """Test instantiation."""
    #         track = StreamTrack(self.interps, stream_data=self.data,
    #                             origin=self.origin, frame=frame)
    #
    #         assert hasattr(track, "_data")
    #         assert hasattr(track, "_track")
    #         assert hasattr(track, "origin")
    #
    #         # --------------
    #         # Different argument types
    #
    #         # The data is an ICRS object
    #         # we must also test passing in a BaseRepresentation
    #         rep = self.data.represent_as(coord.SphericalRepresentation)
    #
    #         track = track_cls(self.interps, stream_data=rep, origin=self.origin)
    #         assert isinstance(track._data_frame, coord.BaseCoordinateFrame)
    #         assert track._data_rep == self.data.representation_type
    #
    #         # and a failed input type
    #         with pytest.raises(TypeError) as e:
    #             track_cls(None, None, None)
    #
    #         assert f"`stream_data` type <{type(None)}> is wrong." in str(e.value)

    def test_path(self, track):
        assert track.path is track._path

    def test_track(self, track):
        assert track.track is track.path.data

    def test_affine(self, track):
        assert track.affine is track.path.affine

    def test_stream_data(self, track):
        assert track.stream_data is track._stream_data

    def test_origin(self, track):
        assert track.origin is track._origin

    def test_frame(self, track):
        assert track.frame is track._path.frame

    def test_frame_fit(self, track):
        if "__attributes__" in track.meta and "frame_fit" in track.meta["__attributes__"]:
            assert track.frame_fit is track.meta["__attributes__"]["frame_fit"]
        else:
            assert track.frame_fit is None

    def test_visit_order(self, track):
        if "__attributes__" in track.meta and "visit_order" in track.meta["__attributes__"]:
            assert track.visit_order is track.meta["__attributes__"]["visit_order"]
        else:
            assert track.visit_order is None

    def test_som(self, track):
        if "__attributes__" in track.meta and "som" in track.meta["__attributes__"]:
            assert track.som is track.meta["__attributes__"]["som"]
        else:
            assert track.som is None

    def test_kalman(self, track):
        if "__attributes__" in track.meta and "som" in track.meta["__attributes__"]:
            assert track.kalman is track.meta["__attributes__"]["kalman"]
        else:
            assert track.kalman is None

    def test_call(self, track, scrd, affine):
        """Test call method."""
        mean, width = track(affine)

        assert isinstance(mean.frame, coord.ICRS)
        assert mean.representation_type == coord.SphericalRepresentation
        assert_quantity_allclose(mean.ra, scrd.ra, atol=1e-15 * u.deg)
        assert_quantity_allclose(mean.dec, scrd.dec, atol=1e-15 * u.deg)
        assert_quantity_allclose(mean.distance, scrd.distance, atol=1e-15 * u.kpc)

    def test_repr(self, track):
        """Test that the modified __repr__ method works."""
        s = track.__repr__()

        frame_name = track.frame.__class__.__name__
        rep_name = track.track.representation_type.__name__
        assert f"StreamTrack ({frame_name}|{rep_name})" in s
