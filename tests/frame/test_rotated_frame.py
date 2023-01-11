# """Testing :mod:`~trackstream.rotated_frame`.

# .. todo::

#     properly use pytest fixtures

# """

#     "test_cartesian_model",
#     "test_residual",


# ##############################################################################
# # IMPORTS

# # THIRD PARTY

# # LOCAL

# ##############################################################################
# # TESTS
# ##############################################################################


# @pytest.mark.parametrize(
#     "test_data,lon,lat,rotation,deg,expected_data",
#             data.icrs,  # TODO as fixture
#             example_coords.RA,
#             example_coords.DEC,
#             example_coords.ICRS_ROTATION,
#             True,
#             data.ricrs,  # TODO as fixture
#         ),
#             data.icrs,  # TODO fixtures
#             example_coords.RA,
#             example_coords.DEC,
#             example_coords.ICRS_ROTATION,
#             False,
#             data.ricrs,  # TODO as fixture
#         ),
#             data.gcentric,  # TODO as fixture
#             example_coords.LON,
#             example_coords.LAT,
#             example_coords.GALACTOCENTRIC_ROTATION,
#             True,
#             data.rgcentric,  # TODO as fixture
#         ),
#             data.gcentric,  # TODO as fixture
#             example_coords.LON,
#             example_coords.LAT,
#             example_coords.GALACTOCENTRIC_ROTATION,
#             False,
#             data.rgcentric,  # TODO as fixture
#         ),
#         # TODO the other datasets
#     ],
# def test_cartesian_model(
#     test_data,
#     lon,
#     lat,
#     rotation,
#     deg: bool,
#     expected_data,
# ):
#     """Test `~trackstream.rotated_frame.cartesian_model`."""
#     # --------------
#     # setup


#     # --------------
#     # apply model

#         test_data.cartesian,
#     # longitude and latitude

#     # --------------
#     # Testing r, lon, lat

#     assert_quantity_allclose(
#         lon,
#     # there's 1 that's almost 180 deg, but not quite and needs to get
#     # modded down.
#     assert_quantity_allclose(


# class TestRotatedFrameFitter:
#     def setup_class(self):

#         # origin

#         # icrs

#         self.RFF = rotated_frame.RotatedFrameFitter(
#             self.data,

#     def test_default_fit_options(self):
#         """Test ``RotatedFrameFitter.default_fit_options``."""

#         assert (set(got.keys()) == set(expect.keys())) & all(

#     def test_set_bounds(self):
#         """Test ``RotatedFrameFitter.set_bounds``."""
#         # original value


#         # back to old value

#     @pytest.mark.skip("TODO!")
#     def test_align_v_positive_lon(self):
#         """Test ``RotatedFrameFitter.align_v_positive_lon``."""
#         assert False

#     @pytest.mark.parametrize(
#         "rotation,expected_lat",
#     def test_residual_scalar(self, rotation, expected_lat):
#         """Test ``RotatedFrameFitter.residual``."""
#         # compare result and expected latitude

#     @pytest.mark.parametrize(
#         "rotation,expected_lat",
#     def test_residual_array(self, rotation, expected_lat):
#         """Test ``RotatedFrameFitter.residual``."""
#         # compare result and expected latitude

#     # ---------------------------------------------------------------

#     def test_fit_no_rot0(self):
#         """Test ``fit`` without ``rot0`` specified."""
#         with pytest.raises(ValueError, match="no prespecified `rot0`"):

#     def test_fit_has_rot0(self):
#         """
#         Test ``fit`` with ``rot0`` specified and all else defaults.
#         This triggers all the ``if X is None`` checks.
#         """
#         assert fr.fitresult.success

#     # ---------------------------------------------------------------


# # -------------------------------------------------------------------


# @pytest.mark.parametrize(
#     "test_data,variables,scalar,expected_lat",
#             data.icrs,  # TODO fixtures
#                 example_coords.ICRS_ROTATION,
#                 example_coords.RA,
#                 example_coords.DEC,
#             ),
#             True,
#             0,  # TODO fixtures
#         ),
#             data.icrs,  # TODO fixtures
#                 example_coords.ICRS_ROTATION,
#                 example_coords.RA,
#                 example_coords.DEC,
#             ),
#             False,
#             data.ricrs.phi2,  # TODO fixtures
#         ),
#             data.gcentric,  # TODO fixtures
#                 example_coords.GALACTOCENTRIC_ROTATION,
#                 example_coords.LON,
#                 example_coords.LAT,
#             ),
#             True,
#             0,  # TODO fixtures
#         ),
#             data.gcentric,  # TODO fixtures
#                 example_coords.GALACTOCENTRIC_ROTATION,
#                 example_coords.LON,
#                 example_coords.LAT,
#             ),
#             False,
#             data.rgcentric.phi2,  # TODO fixtures
#         ),
#         # TODO the other datasets
#     ],
# def test_residual(test_data, variables, scalar, expected_lat):
#     """Test `~trackstream.rotated_frame.residual`."""

#     # compare result and expected latitude
#     # When scalar is False, can have units, so always take value.
#     np.testing.assert_allclose(
#         res,
