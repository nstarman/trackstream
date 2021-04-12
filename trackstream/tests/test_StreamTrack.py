# -*- coding: utf-8 -*-

"""Testing :class:`trackstream.core.StreamTrack`."""

__all__ = [
    "Test_StreamTrack",
]


##############################################################################
# IMPORTS

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import pytest
from astropy.tests.helper import assert_quantity_allclose

# PROJECT-SPECIFIC
from trackstream import StreamTrack
from trackstream.tests.helper import BaseClassDependentTests
from trackstream.utils import interpolated_coordinates as icoord
from trackstream.utils.path import Path

##############################################################################
# PARAMETERS


##############################################################################
# TESTS
##############################################################################


class Test_StreamTrack(BaseClassDependentTests, klass=StreamTrack):
    """Test :class:`~trackstream.core.StreamTrack`."""

    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        num = 40
        cls.arclength = np.linspace(0, 10, num=num) * u.deg

        lon = np.linspace(0, 25, num=num) * u.deg
        lat = np.linspace(-10, 10, num=num) * u.deg
        distance = np.linspace(8, 15, num=num) * u.kpc
        width = 100 * u.pc

        cls.rep = coord.SphericalRepresentation(
            lon=lon,
            lat=lat,
            distance=distance,
        )

        cls.frame = coord.ICRS()
        cls.data = icoord.InterpolatedCoordinateFrame(
            cls.frame.realize_frame(cls.rep),
            affine=cls.arclength,
        )
        cls.path = Path(cls.data, width=width)

        # origin
        i = num // 2
        cls.origin = coord.ICRS(ra=lon[i], dec=lat[i])

        # track
        cls.track = StreamTrack(
            cls.path,
            stream_data=cls.data,
            origin=cls.origin,
            frame=cls.frame,
        )

    # /def

    # -------------------------------

    def test_instantiation(self):
        """Test instantiation."""
        track = StreamTrack(
            self.path,
            stream_data=self.data,
            origin=self.origin,
            frame=self.frame,
        )

        assert hasattr(track, "_stream_data")
        assert hasattr(track, "_path")
        assert hasattr(track, "_origin")
        assert hasattr(track, "_frame")

        # --------------
        # Different argument types

        # The data is an ICRS object
        # we must also test passing in a BaseRepresentation
        rep = self.data.represent_as(coord.SphericalRepresentation)

        track = self.klass(
            self.path,
            stream_data=rep,
            origin=self.origin,
            frame=self.frame,
        )
        assert isinstance(track.frame, coord.BaseCoordinateFrame)
        assert track.track.representation_type == self.data.representation_type

        # and a failed input type
        with pytest.raises(TypeError) as e:
            self.klass(None, stream_data=None, origin=None, frame=None)
        assert "`path` must be" in str(e.value)

        with pytest.raises(TypeError) as e:
            self.klass(self.path, stream_data=None, origin=None, frame=None)
        assert "`origin` must be " in str(e.value)

    # /def

    def test_call(self):
        """Test call method."""
        data, width = self.track(self.arclength)

        assert isinstance(data.frame, coord.ICRS)
        assert data.representation_type == coord.SphericalRepresentation
        assert_quantity_allclose(data.ra, self.data.ra, atol=1e-15 * u.deg)
        assert_quantity_allclose(data.dec, self.data.dec, atol=1e-15 * u.deg)
        assert_quantity_allclose(
            data.distance,
            self.data.distance,
            atol=1e-15 * u.kpc,
        )

    # /def

    def test_repr(self):
        """Test that the modified __repr__ method works."""
        s = self.track.__repr__()

        frame_name = self.track.frame.__class__.__name__
        rep_name = self.track.path.path.representation_type.__name__
        assert f"StreamTrack ({frame_name}|{rep_name})" in s

    # /def

    # -------------------------------


# /class


# -------------------------------------------------------------------


##############################################################################
# END
