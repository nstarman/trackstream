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

# LOCAL
from trackstream import StreamTrack
from trackstream.tests.helper import BaseClassDependentTests
from trackstream.utils import InterpolatedUnivariateSplinewithUnits as IUSU

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

        cls.data = coord.ICRS(
            coord.SphericalRepresentation(lon=lon, lat=lat, distance=distance),
        )
        cls.interps = dict(
            lon=IUSU(cls.arclength, lon),
            lat=IUSU(cls.arclength, lat),
            distance=IUSU(cls.arclength, distance),
        )

        # origin
        i = num // 2
        cls.origin = coord.ICRS(ra=lon[i], dec=lat[i])

        # track
        cls.track = StreamTrack(
            cls.interps,
            stream_data=cls.data,
            origin=cls.origin,
        )

    # /def

    # -------------------------------

    def test_instantiation(self):
        """Test instantiation."""
        track = StreamTrack(
            self.interps,
            stream_data=self.data,
            origin=self.origin,
        )

        assert hasattr(track, "_data")
        assert hasattr(track, "_track")
        assert hasattr(track, "origin")

        # --------------
        # Different argument types

        # The data is an ICRS object
        # we must also test passing in a BaseRepresentation
        rep = self.data.represent_as(coord.SphericalRepresentation)

        track = self.klass(self.interps, stream_data=rep, origin=self.origin)
        assert isinstance(track._data_frame, coord.BaseCoordinateFrame)
        assert track._data_rep == self.data.representation_type

        # and a failed input type
        with pytest.raises(TypeError) as e:
            self.klass(None, None, None)

        assert f"`stream_data` type <{type(None)}> is wrong." in str(e.value)

    # /def

    def test_call(self):
        """Test call method."""
        data = self.track(self.arclength)

        assert isinstance(data, coord.ICRS)
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

        frame_name = self.track._data_frame.__class__.__name__
        rep_name = self.track._data_rep.__name__
        assert f"StreamTrack ({frame_name}|{rep_name})" in s

    # /def

    # -------------------------------


# /class


# -------------------------------------------------------------------


##############################################################################
# END
