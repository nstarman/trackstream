# -*- coding: utf-8 -*-

"""Testing :class:`trackstream.core.TrackStream`."""

__all__ = [
    "Test_TrackStream",
]


##############################################################################
# IMPORTS

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import pytest
from astropy.table import QTable

# LOCAL
from trackstream import TrackStream
from trackstream.tests.helper import BaseClassDependentTests

##############################################################################
# PARAMETERS


##############################################################################
# TESTS
##############################################################################


class Test_TrackStream(BaseClassDependentTests, klass=TrackStream):
    """Test :class:`~trackstream.core.TrackStream`."""

    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        # Make Data in rotated frame, to ensure linearity
        num = 40
        lon = np.linspace(0, 25, num=num) * u.deg
        lat = np.linspace(0, 0, num=num) * u.deg
        distance = np.linspace(8, 15, num=num) * u.kpc

        i = num // 2  # index of origin
        cls.origin = coord.ICRS(ra=lon[i], dec=lat[i], distance=distance[i])
        cls.rotframe = coord.SkyOffsetFrame(
            origin=cls.origin,
            rotation=-37 * u.deg,
        )
        rotdata = cls.rotframe.realize_frame(
            coord.SphericalRepresentation(lon=lon, lat=lat, distance=distance),
        )

        # given the construction, the lon variable is a good arclength,
        # just need to adjust for the origin
        cls.arclength = lon - lon[i]

        # make data in ICRS from rotated data
        cls.data = rotdata.transform_to(coord.ICRS)
        cls.lon = cls.data.ra
        cls.lat = cls.data.dec
        cls.distance = cls.data.distance

        cls.data_err = QTable(
            cls.data.cartesian.xyz.T / 10,
            names=["x_err", "y_err", "z_err"],
        )

        cls.tracker = cls.klass(
            data=cls.data,
            origin=cls.origin,
            data_err=cls.data_err,
            frame=cls.rotframe,
        )

    # /def

    # -------------------------------

    def test_instantiation(self):
        """Test instantiation."""
        # --------------
        # Different levels of arguments

        # Start with the minimal
        self.klass(self.data, self.origin)
        # non-zero data errors
        self.klass(self.data, self.origin, data_err=self.data_err)
        # info on frame rotation
        self.klass(
            self.data,
            self.origin,
            data_err=self.data_err,
            frame=self.rotframe,
        )

        # TODO SOM

        # --------------
        # Different argument types

        # The data is an ICRS object
        # we must also test passing in a BaseRepresentation
        rep = self.data.represent_as(coord.SphericalRepresentation)

        tracker = self.klass(rep, self.origin)
        assert isinstance(tracker._data_frame, coord.BaseCoordinateFrame)
        assert tracker._data_rep == self.data.representation_type

        # and a failed input type
        with pytest.raises(TypeError) as e:
            self.klass(None, self.origin)

        assert f"`data` type <{type(None)}> is wrong." in str(e.value)

        # problem with contents of data_err
        # right type, wrong column names
        with pytest.raises(ValueError):
            self.klass(
                self.data,
                self.origin,
                data_err=QTable(
                    self.data.cartesian.xyz.T / 10,
                    names=["d_x", "d_y", "d_z"],
                ),
            )

    # /def

    # -------------------------------

    def test_fit(self):
        """Test method ``fit``."""
        track = self.tracker.fit()

        with pytest.raises(AttributeError):  # can't call what don't have
            track(None)

    # /def

    def test_predict(self):
        """Test method ``predict``."""
        with pytest.raises(AttributeError):  # can't call what don't have
            self.tracker.predict(self.arclength)

    # /def

    def test_fit_predict(self):
        """Test method ``fit_predict``."""
        with pytest.raises(AttributeError):  # can't call what don't have
            self.tracker.fit_predict(self.arclength)

    # /def


# /class


# -------------------------------------------------------------------


##############################################################################
# END
