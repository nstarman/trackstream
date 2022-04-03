# -*- coding: utf-8 -*-

"""Testing :mod:`~trackstream.utils.coord_utils`."""


##############################################################################
# IMPORTS

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import pytest

# LOCAL
from trackstream.config import conf
from trackstream.utils.coord_utils import (  # reference_to_skyoffset_matrix,
    cartesian_to_spherical,
    resolve_framelike,
)

##############################################################################
# TESTS
##############################################################################


@pytest.mark.parametrize("x, y, z", [(1, 2, 3), (10, -12, 45)])
def test_cartesian_to_spherical(x, y, z):
    # expect
    c = coord.CartesianRepresentation(x, y, z).represent_as(coord.SphericalRepresentation)

    # got
    lon, lat, r = cartesian_to_spherical(x, y, z, deg=False)
    assert r == c.distance
    assert np.allclose(lat, c.lat.to_value(u.rad))
    assert np.allclose(lon % (2 * np.pi), c.lon.to_value(u.rad))

    lon, lat, r = cartesian_to_spherical(x, y, z, deg=True)
    assert r == c.distance
    assert np.allclose(lat, c.lat.to_value(u.deg))
    assert np.allclose(lon % (360), c.lon.to_value(u.deg))


@pytest.mark.skip("Modified from astropy. Don't really need to test.")
@pytest.mark.parametrize("lon, lat, rotation", [(1, 2, 100), (10, -12, 45 * u.deg)])
def test_reference_to_skyoffset_matrix(lon, lat, rotation):
    """Test `reference_to_skyoffset_matrix`."""
    pass


@pytest.mark.parametrize("error_if_not_type", [True, False])
class Test_resolve_framelike:
    """Test ``resolve_framelike``. Uses :func:`functools.singledispatch`."""

    def test_None(self, error_if_not_type):
        frame = resolve_framelike(None, error_if_not_type=error_if_not_type)
        assert frame.__class__.__name__.lower() == conf.default_frame

    def test_str(self, error_if_not_type):
        frame = resolve_framelike("galactic", error_if_not_type=error_if_not_type)
        assert isinstance(frame, coord.Galactic)

    def test_coordframe(self, error_if_not_type):
        frame = resolve_framelike(coord.Galactocentric(), error_if_not_type=error_if_not_type)
        assert isinstance(frame, coord.Galactocentric)

    def test_skycoord(self, error_if_not_type):
        c = coord.ICRS(ra=1 * u.deg, dec=2 * u.deg)
        frame = resolve_framelike(coord.SkyCoord(c))
        assert isinstance(frame, coord.ICRS)

    def test_wrong_type(self, error_if_not_type):
        if not error_if_not_type:
            frame = resolve_framelike(object(), error_if_not_type=error_if_not_type)
            assert frame.__class__.__name__ == "object"

        else:
            with pytest.raises(TypeError):
                resolve_framelike(object(), error_if_not_type=error_if_not_type)
