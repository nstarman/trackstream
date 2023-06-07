# ##############################################################################
# # IMPORTS

# # THIRD PARTY

# # LOCAL

# ##############################################################################
# # TESTS
# ##############################################################################


# class Test_FitterStreamArmTrack:

#     def setup_class(self):
#         """Setup fixtures for testing."""

#     @pytest.fixture(params=[(None, None), (True, None), (None, True), (True, True)])
#     def tracker(self, request):
#         if arm1SOM is True:
#             # TODO: have a test for a pre-made SOM
#         if arm2SOM is True:
#             # TODO: have a test for a pre-made SOM


#     @pytest.fixture
#     def tracker_cls(self):

#     @pytest.fixture
#     def stream_cls(self):
#         """Stream class."""

#     @pytest.fixture
#     def stream(self, stream_cls):
#         """Stream instance."""

#     # Method tests

#     def test_init(self, tracker):
#         """Test instantiation."""

#     # -------------------------------

#     def test_fit(self, tracker, stream):
#         """Test method ``fit``."""

#         # TODO: a lot more tests

#     def test_predict(self, tracker):
#         """Test method ``predict``."""
#         with pytest.raises(AttributeError):  # can't call what don't have

#     def test_fit_predict(self, tracker, stream):
#         """Test method ``fit_predict``."""
#         with pytest.raises(AttributeError):  # can't call what don't have


# # /class


# # -------------------------------------------------------------------


# class Test_StreamArmTrack:

#     #     @classmethod
#     #         """Setup fixtures for testing."""
#     #         # TODO: move to
#     #
#     #
#     #
#     #         # origin

#     @pytest.fixture
#     def origin(self, scrd, num):

#     @pytest.fixture
#     def track_cls(self):

#     @pytest.fixture
#     def track(self, track_cls, path, scrd, origin, frame):
#         """path and stream_data don't have to match up this nicely."""

#     # Method tests

#     #         """Test instantiation."""
#     #
#     #
#     #         # --------------
#     #         # Different argument types
#     #
#     #         # The data is an ICRS object
#     #         # we must also test passing in a BaseRepresentation
#     #
#     #
#     #         # and a failed input type
#     #

#     def test_path(self, track):
#         assert track.path is track._path

#     def test_track(self, track):
#         assert track.track is track.path.data

#     def test_affine(self, track):
#         assert track.affine is track.path.affine

#     def test_stream_data(self, track):
#         assert track.stream_data is track._stream_data

#     def test_origin(self, track):
#         assert track.origin is track._origin

#     def test_frame(self, track):
#         assert track.frame is track._path.frame

#     def test_frame_fit(self, track):
#         if "__attributes__" in track.meta and "frame_fit" in track.meta["__attributes__"]:
#             assert track.frame_fit is None

#     def test_visit_order(self, track):
#         if "__attributes__" in track.meta and "visit_order" in track.meta["__attributes__"]:
#             assert track.visit_order is None

#     def test_som(self, track):
#         if "__attributes__" in track.meta and "som" in track.meta["__attributes__"]:
#             assert track.som is None

#     def test_kalman(self, track):
#         if "__attributes__" in track.meta and "som" in track.meta["__attributes__"]:
#             assert track.kalman is None

#     def test_call(self, track, scrd, affine):
#         """Test call method."""


#     def test_repr(self, track):
#         """Test that the modified __repr__ method works."""
