# -*- coding: utf-8 -*-

"""Testing :mod:`~trackstream.utils._framelike`."""

__all__ = [
    "test_resolve_framelike",
]


##############################################################################
# IMPORTS

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import pytest

# PROJECT-SPECIFIC
from trackstream.config import conf
from trackstream.utils import _framelike as framelike

##############################################################################
# TESTS
##############################################################################


def test_resolve_framelike():
    """Test ``test_resolve_framelike``."""
    # -----------------------
    # default

    frame = framelike.resolve_framelike(None)

    assert frame.__class__.__name__.lower() == conf.default_frame

    # -----------------------
    # str

    frame = framelike.resolve_framelike("galactic")

    assert frame == coord.Galactic()

    # -----------------------
    # BaseCoordinateFrame

    frame = framelike.resolve_framelike(coord.Galactocentric())

    assert frame == coord.Galactocentric()

    # -----------------------
    # SkyCoord

    frame = framelike.resolve_framelike(
        coord.SkyCoord(coord.ICRS(ra=1 * u.deg, dec=2 * u.deg)),
    )

    assert frame == coord.ICRS()

    # -----------------------
    # error if wrong type
    # error_if_not_type is default True

    with pytest.raises(TypeError):
        framelike.resolve_framelike(object())

    with pytest.raises(TypeError):
        framelike.resolve_framelike(object(), error_if_not_type=True)

    # -----------------------
    # pass thru if flag is False

    frame = framelike.resolve_framelike(object(), error_if_not_type=False)

    assert frame.__class__.__name__ == "object"


# /def

# -------------------------------------------------------------------


##############################################################################
# END
