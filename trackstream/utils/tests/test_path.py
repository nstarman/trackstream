# -*- coding: utf-8 -*-

"""Testing :mod:`trackstream.utils.path`."""


##############################################################################
# IMPORTS

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import pytest

# LOCAL
import trackstream.utils.interpolated_coordinates as icoord
from trackstream.utils.path import Path, path_moments

##############################################################################
# TESTS
##############################################################################


class TestPath:
    """Test :class:`trackstream.utils.path.Path`."""

    # ===============================================================
    # Method tests

    def test_init(self, path_cls, iscrd, width, affine, frame):
        """Test initialization."""
        path = path_cls(iscrd, width, name="test_init", affine=affine, frame=frame)
        assert hasattr(path, "_name")
        assert hasattr(path, "_frame")
        assert hasattr(path, "_original_path")
        assert hasattr(path, "_iscrd")

        # TODO! tests for initialize width

    def test_name(self, path):
        """Test :attr:`trackstream.utils.path.Path.name`."""
        assert path.name is path._name

    def test_frame(self, path):
        """Test :attr:`trackstream.utils.path.Path.frame`."""
        assert path.frame is path._frame

    def test_data(self, path):
        """Test :attr:`trackstream.utils.path.Path.path`."""
        assert path.data is path._iscrd

    def test_affine(self, path, affine):
        """Test :attr:`trackstream.utils.path.Path.affine`."""
        assert path.affine is path.data.affine
        assert np.all(path.affine == affine)

    def test_width(self, path, affine):
        """Test :meth:`trackstream.utils.path.Path.width`."""
        # default
        assert np.array_equal(path.width(None), path.width(affine))

        i = len(affine) // 2
        assert u.allclose(path.width(affine[i]), 100 * u.pc)

        # TODO! test non-constant function

    def test_call(self, path, affine):
        # default
        got = path()

        assert isinstance(got, path_moments)
        assert np.all(got.mean == path.data())
        assert np.all(got.width == path.width())

        # scalar
        i = len(affine) // 2
        got = path(affine[i])
        assert got.mean.separation_3d(path.data[i]) < 1 * u.pc
        assert np.all(got.width == path.width(affine[i]))

        # interpolation
        got = path(5 * u.deg)
        expected_mean = coord.SkyCoord(
            ra=71.56505118 * u.deg,
            dec=57.68846676 * u.deg,
            distance=2.95803989 * u.kpc,
            pm_ra_cosdec=-0.13530872 * u.mas / u.yr,
            pm_dec=-0.11435674 * u.mas / u.yr,
            radial_velocity=7.52187287 * u.km / u.s,
        )
        assert got.mean.separation_3d(expected_mean) < 1 * u.pc
        assert u.allclose(got.width, 100 * u.pc)


##############################################################################
# END
