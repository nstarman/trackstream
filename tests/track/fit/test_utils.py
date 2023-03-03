"""Testing :mod:`~trackstream.utils.coord_utils`."""

from math import pi

import astropy.coordinates as coord
from astropy.coordinates import (
    FK5,
    LSR,
    Angle,
    CartesianDifferential,
    CartesianRepresentation,
    SphericalDifferential,
    SphericalRepresentation,
)
import astropy.units as u
from astropy.units import Quantity
import numpy as np
from numpy import allclose, array, array_equal
import pytest

from trackstream.track.fit.utils import offset_by, position_angle
from trackstream.track.utils import covariance_ellipse, is_structured
from trackstream.utils.coord_utils import deep_transform_to, parse_framelike

##############################################################################
# TESTS
##############################################################################


@pytest.mark.skip("Modified from astropy. Don't really need to test.")
@pytest.mark.parametrize(("lon", "lat", "rotation"), [(1, 2, 100), (10, -12, Quantity(45, u.deg))])
def test_reference_to_skyoffset_matrix(lon, lat, rotation):  # noqa: ARG001
    """Test :func:`trackstream.utils.coord_utils.reference_to_skyoffset_matrix`."""
    assert True


class Test_parse_framelike:
    """Test :func:`trackstream.utils.coord_utils.parse_framelike`.

    Uses :func:`functools.singledispatch`.
    """

    def test_wrong_type(self):
        """Test giving the wrong type to ``parse_framelike``."""
        with pytest.raises(NotImplementedError, match="frame type"):
            parse_framelike(object())

    def test_str(self):
        """Test ``resolve_framelik`` with `str` input."""
        frame = parse_framelike("galactic")
        assert isinstance(frame, coord.Galactic)

    def test_frame(self):
        """Test ``resolve_framelik`` with |Frame| input."""
        frame = parse_framelike(coord.Galactocentric())
        assert isinstance(frame, coord.Galactocentric)


@pytest.mark.parametrize(
    ("reptype", "diftype"),
    [
        (CartesianRepresentation, CartesianDifferential),
        (SphericalRepresentation, SphericalDifferential),
        (CartesianRepresentation, "base"),
        (CartesianRepresentation, None),
    ],
)
@pytest.mark.parametrize("output_frame", [FK5, LSR])
def test_deep_transform_to(crd, frame, output_frame, reptype, diftype):
    """Test :func:`trackstream.utils.coord_utils.deep_transform_to`."""
    c = deep_transform_to(crd, frame=output_frame, representation_type=reptype, differential_type=diftype)

    frame = c.frame if hasattr(c, "frame") else c

    assert isinstance(frame, output_frame)
    assert frame.representation_type is reptype

    if diftype in ("base", None):
        pass  # it's inferred
    else:
        assert frame.differential_type is diftype

    data = frame.data
    assert isinstance(data, reptype)
    if diftype is None:
        assert "s" not in data.differentials
    elif diftype == "base":
        assert "s" in data.differentials
    else:
        assert isinstance(data.differentials["s"], diftype)


@pytest.mark.parametrize(
    ("lon1", "lat1", "lon2", "lat2", "expected"),
    [
        (0, 0, pi, 0, pi / 2),
        (0, 0, 0, pi, 0),
        (array([0, 0]), array([0, 0]), array([pi, 0]), array([0, pi]), array([pi / 2, 0])),
    ],
)
def test_position_angle(lon1, lat1, lon2, lat2, expected):
    """Test `trackstream.utils.coord_utils.position_angle`."""
    pa = position_angle(lon1, lat1, lon2, lat2)
    assert array_equal(pa, expected)


@pytest.mark.parametrize(
    ("lon", "lat", "posang", "distance", "expected"),
    [(0, 0, pi, 1, (0, -1)), (0, 0, 0, pi, (pi, 0)), (0, 0, [pi, 0], [1, pi], ([0, pi], [-1, 0]))],
)
def test_offset_by(lon, lat, posang, distance, expected):
    """Test `trackstream.utils.coord_utils.offset_by`."""
    pnt = offset_by(lon, lat, posang, distance)
    assert allclose(pnt, expected)


def test_is_structured():
    """Test :func:`trackstream.utils.misc.is_structured`."""
    # Not structured
    assert not is_structured(None)
    assert not is_structured(np.array(1))

    # structured
    assert is_structured(np.array((1, 2), dtype=[("f1", int), ("f2", int)]))


@pytest.mark.parametrize(
    ("P", "nstd", "expected"),
    [
        ([[1, 0], [0, 1]], 1, (0, [1, 1])),
        ([[1, 0], [0, 1]], 2, (0, [2, 2])),
        ([[0, 1], [1, 0]], 1, (Angle(-90, u.deg), [1, 1])),
        ([[0, 1], [1, 0]], 2, (Angle(-90, u.deg), [2, 2])),
    ],
)
def test_covariance_ellipse(P, nstd, expected):
    """Test :func:`trackstream.utils.misc.covariance_ellipse`."""
    (orientation, wh) = covariance_ellipse(P, nstd=nstd)
    assert orientation == expected[0]
    assert np.array_equal(wh, expected[1])
