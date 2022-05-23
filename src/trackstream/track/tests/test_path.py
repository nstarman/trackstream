# -*- coding: utf-8 -*-

# """Testing :mod:`trackstream.utils.path`."""


# ##############################################################################
# # IMPORTS

# # STDLIB
# from typing import Any, Callable, Optional, Union

# # THIRD PARTY
# import astropy.coordinates as coord
# import astropy.units as u
# import interpolated_coordinates as icoord
# import numpy as np
# import pytest

# # LOCAL
# from trackstream.utils.path import Path, path_moments

# ##############################################################################
# # TESTS
# ##############################################################################


# def test_path_moments():
#     """Test :class:`trackstream.utils.path.path_moments`."""
#     # Check it's a tuple
#     assert issubclass(path_moments, tuple)

#     # Check fields
#     assert path_moments._fields == ("mean", "width")
#     assert path_moments.__annotations__["mean"] == coord.SkyCoord
#     assert path_moments.__annotations__["width"] == u.Quantity


# class TestPath:
#     """Test :class:`trackstream.utils.path.Path`."""

#     # ===============================================================
#     # Method and Attribute tests

#     def test_annotations(self, path_cls):
#         """Test has expected type annotations."""
#         annot = path_cls.__annotations__

#         assert annot["_name"] is Optional[str]
#         assert annot["_frame"] is coord.BaseCoordinateFrame
#         assert annot["_original_path"] is Any
#         assert annot["_iscrd"] is icoord.InterpolatedSkyCoord
#         assert annot["_original_width"] is Union[u.Quantity, Callable, None]
#         assert annot["_width_fn"] is Optional[Callable]

#     # -----------------------------------------------------

#     def test_init_default(self, path_cls, iscrd, width, affine, frame):
#         """Test initialization."""
#         # Standard
#         path = path_cls(iscrd, width, name="test_init", affine=affine, frame=frame)

#         assert isinstance(path, Path)
#         assert hasattr(path, "_name")
#         assert hasattr(path, "_frame")
#         assert hasattr(path, "_original_path")
#         assert hasattr(path, "_iscrd")

#         assert path.name == "test_init"
#         assert path.frame == frame
#         assert np.allclose(path.data.separation_3d(iscrd.transform_to(frame),
#                            interpolate=False), 0)
#         assert np.array_equal(path.affine, affine)

#     def test_init_frame_and_data(self, path_cls, iscrd, width, affine, frame):
#         """Test initialization with different frames and data types."""
#         # Frame = None, path is SkyCoord-like
#         path = path_cls(iscrd, width, name="test_init", affine=affine, frame=None)
#         assert path.frame == iscrd.frame.replicate_without_data()
#         assert isinstance(path.data, icoord.InterpolatedSkyCoord)

#         # Frame = None, path is Coordinate-like
#         path = path_cls(iscrd.frame, width, name="test_init", affine=affine, frame=None)
#         assert path.frame == iscrd.frame.replicate_without_data()
#         assert isinstance(path.data, icoord.InterpolatedSkyCoord)

#     @pytest.mark.skip("TODO!")
#     def test_init_width(self):
#         assert False
#         # TODO! tests for initialize width

#     # -----------------------------------------------------

#     def test_name(self, path):
#         """Test :attr:`trackstream.utils.path.Path.name`."""
#         assert path.name is path._name

#     def test_frame(self, path):
#         """Test :attr:`trackstream.utils.path.Path.frame`."""
#         assert path.frame is path._frame

#     def test_data(self, path):
#         """Test :attr:`trackstream.utils.path.Path.path`."""
#         assert path.data is path._iscrd

#     def test_affine(self, path, affine):
#         """Test :attr:`trackstream.utils.path.Path.affine`."""
#         assert path.affine is path.data.affine
#         assert np.all(path.affine == affine)

#     # -----------------------------------------------------

#     def test_call(self, path, affine):
#         """Test :meth:`trackstream.utils.path.Path.__call__`."""
#         # default
#         got = path()

#         assert isinstance(got, path_moments)
#         assert np.all(got.mean == path.data())
#         assert np.all(got.width == path.width())

#         # scalar
#         i = len(affine) // 2
#         got = path(affine[i])
#         assert got.mean.separation_3d(path.data[i]) < 1 * u.pc
#         assert np.all(got.width == path.width(affine[i]))

#         # interpolation
#         got = path(5 * u.deg)
#         expected_mean = coord.SkyCoord(
#             ra=71.56505118 * u.deg,
#             dec=57.68846676 * u.deg,
#             distance=2.95803989 * u.kpc,
#             pm_ra_cosdec=-0.13530872 * u.mas / u.yr,
#             pm_dec=-0.11435674 * u.mas / u.yr,
#             radial_velocity=7.52187287 * u.km / u.s,
#         )
#         assert got.mean.separation_3d(expected_mean) < 1 * u.pc
#         assert u.allclose(got.width, 100 * u.pc)

#     def test_position(self, path, affine):
#         """Test :meth:`trackstream.utils.path.Path.position`.

#         This calls `trackstream.utils.interpolated_coordinates.InterpolatedSkyCoord`,
#         so we delegate most of the tests to there, here only checking the
#         results are as expected.
#         """
#         # default
#         assert np.allclose(path.position(None).separation_3d(path.position(affine)), 0)

#         # an actual interpolated position
#         got = path.position(5 * u.deg)
#         expected_mean = coord.SkyCoord(
#             ra=71.56505118 * u.deg,
#             dec=57.68846676 * u.deg,
#             distance=2.95803989 * u.kpc,
#             pm_ra_cosdec=-0.13530872 * u.mas / u.yr,
#             pm_dec=-0.11435674 * u.mas / u.yr,
#             radial_velocity=7.52187287 * u.km / u.s,
#         )
#         assert np.all(got.separation_3d(expected_mean) < 0.1 * u.pc)

#     def test_width(self, path, affine):
#         """Test :meth:`trackstream.utils.path.Path.width`."""
#         # default
#         assert np.array_equal(path.width(None), path.width(affine))

#         i = len(affine) // 2
#         assert u.allclose(path.width(affine[i]), 100 * u.pc)

#         # TODO! test non-constant function

#     @pytest.mark.skip("TODO!")
#     def test_width_angular(self, path, affine):
#         """Test :meth:`trackstream.utils.path.Path.width_angular`."""
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
#         afn = path.closest_affine_to_point(point_on)
#         assert np.isclose(afn, affine_on)

#         # TODO! a point off the path

#     @pytest.mark.parametrize("angular", [False, True])
#     def test_closest_position_to_point(self, path, point_on, angular, affine_on):
#         p = path.closest_position_to_point(point_on)
#         assert p.separation(point_on) < 1e-8 * u.deg
