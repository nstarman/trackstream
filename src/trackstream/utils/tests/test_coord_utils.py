"""Testing :mod:`~trackstream.utils.coord_utils`."""


##############################################################################
# IMPORTS

# STDLIB
from math import pi

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import pytest
from astropy.coordinates import (
    FK5,
    LSR,
    CartesianDifferential,
    CartesianRepresentation,
    SphericalDifferential,
    SphericalRepresentation,
)
from astropy.units import Quantity
from numpy import allclose, array, array_equal

# LOCAL
from trackstream.utils.coord_utils import (
    deep_transform_to,
    offset_by,
    position_angle,
    resolve_framelike,
)

##############################################################################
# TESTS
##############################################################################


@pytest.mark.skip("Modified from astropy. Don't really need to test.")
@pytest.mark.parametrize("lon, lat, rotation", [(1, 2, 100), (10, -12, Quantity(45, u.deg))])
def test_reference_to_skyoffset_matrix(lon, lat, rotation):
    """Test :func:`trackstream.utils.coord_utils.reference_to_skyoffset_matrix`."""
    assert True


@pytest.mark.parametrize("type_error", [True, False])
class Test_resolve_framelike:
    """Test :func:`trackstream.utils.coord_utils.resolve_framelike`.

    Uses :func:`functools.singledispatch`.
    """

    def test_wrong_type(self, type_error):
        """Test giving the wrong type to ``resolve_framelike``."""
        if not type_error:
            frame = resolve_framelike(object(), type_error=type_error)
            assert frame.__class__.__name__ == "object"

        else:
            with pytest.raises(TypeError, match="input coordinate"):
                resolve_framelike(object(), type_error=type_error)

    def test_str(self, type_error):
        """Test ``resolve_framelik`` with `str` input."""
        frame = resolve_framelike("galactic", type_error=type_error)
        assert isinstance(frame, coord.Galactic)

    def test_frame(self, type_error):
        """Test ``resolve_framelik`` with |Frame| input."""
        frame = resolve_framelike(coord.Galactocentric(), type_error=type_error)
        assert isinstance(frame, coord.Galactocentric)

    def test_skycoord(self, type_error):
        """Test ``resolve_framelik`` with |SkyCoord| input."""
        c = coord.ICRS(ra=Quantity(1, u.deg), dec=Quantity(2, u.deg))
        frame = resolve_framelike(coord.SkyCoord(c), type_error=type_error)
        assert isinstance(frame, coord.ICRS)


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
