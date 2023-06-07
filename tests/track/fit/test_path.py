# ##############################################################################
# # IMPORTS

# # STDLIB

# # THIRD PARTY

# # LOCAL

# ##############################################################################
# # TESTS
# ##############################################################################


# def test_path_moments():
#     # Check it's a tuple

#     # Check fields


# class TestPath:

#     # Method and Attribute tests

#     def test_annotations(self, path_cls):
#         """Test has expected type annotations."""


#     # -----------------------------------------------------

#     def test_init_default(self, path_cls, iscrd, width, affine, frame):
#         """Test initialization."""
#         # Standard


#         assert np.allclose(path.data.separation_3d(iscrd.transform_to(frame),
#                            interpolate=False), 0)

#     def test_init_frame_and_data(self, path_cls, iscrd, width, affine, frame):
#         """Test initialization with different frames and data types."""


#     @pytest.mark.skip("TODO!")
#     def test_init_width(self):
#         assert False
#         # TODO: tests for initialize width

#     # -----------------------------------------------------

#     def test_name(self, path):
#         assert path.name is path._name

#     def test_frame(self, path):
#         assert path.frame is path._frame

#     def test_data(self, path):
#         assert path.data is path._iscrd

#     def test_affine(self, path, affine):
#         assert path.affine is path.data.affine

#     # -----------------------------------------------------

#     def test_call(self, path, affine):
#         # default


#         # scalar

#         # interpolation

#     def test_position(self, path, affine):
#         """Test :meth:`trackstream.utils.path.Path.position`.

#         This calls `trackstream.utils.interpolated_coordinates.InterpolatedSkyCoord`,
#         so we delegate most of the tests to there, here only checking the
#         results are as expected.
#         """
#         # default

#         # an actual interpolated position

#     def test_width(self, path, affine):
#         # default


#         # TODO: test non-constant function

#     @pytest.mark.skip("TODO!")
#     def test_width_angular(self, path, affine):
#         assert False

#     # -----------------------------------------------------

#     @pytest.mark.skip("TODO!")
#     def test_separation(self, path, affine):
#         """Test :meth:`trackstream.utils.path.Path.separation`.

#         This calls `trackstream.utils.interpolated_coordinates.InterpolatedSkyCoord`,
#         so we delegate most of the tests to there, here only checking the
#         results are as expected.
#         """
#         assert False

#     @pytest.mark.skip("TODO!")
#     def test_separation_3d(self, path, affine):
#         """Test :meth:`trackstream.utils.path.Path.separation_3d`.

#         This calls `trackstream.utils.interpolated_coordinates.InterpolatedSkyCoord`,
#         so we delegate most of the tests to there, here only checking the
#         results are as expected.
#         """
#         assert False

#     # -----------------------------------------------------

#     @pytest.mark.parametrize("angular", [False, True])
#     def test_closest_affine_to_point(self, path, point_on, angular, affine_on):

#         # TODO: a point off the path

#     @pytest.mark.parametrize("angular", [False, True])
#     def test_closest_position_to_point(self, path, point_on, angular, affine_on):
