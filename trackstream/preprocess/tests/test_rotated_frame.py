# -*- coding: utf-8 -*-

"""Testing :mod:`~trackstream.preprocess.rotated_frame`.

.. todo::

    properly use pytest fixtures

"""

# __all__ = [
# ]


##############################################################################
# IMPORTS

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import pytest
from astropy.tests.helper import assert_quantity_allclose

# PROJECT-SPECIFIC
from trackstream.example_data import example_coords
from trackstream.example_data.tests import data
from trackstream.preprocess import rotated_frame

##############################################################################
# Fixtures


##############################################################################
# TESTS
##############################################################################


@pytest.mark.parametrize(
    "test_data,lon,lat,rotation,deg,expected_data",
    [
        (
            data.icrs,  # TODO fixtures
            example_coords.RA,
            example_coords.DEC,
            example_coords.ICRS_ROTATION,
            True,
            data.ricrs,  # TODO fixtures
        ),
        (
            data.icrs,  # TODO fixtures
            example_coords.RA,
            example_coords.DEC,
            example_coords.ICRS_ROTATION,
            False,
            data.ricrs,  # TODO fixtures
        ),
        (
            data.gcentric,  # TODO fixtures
            example_coords.LON,
            example_coords.LAT,
            example_coords.GALACTOCENTRIC_ROTATION,
            True,
            data.rgcentric,  # TODO fixtures
        ),
        (
            data.gcentric,  # TODO fixtures
            example_coords.LON,
            example_coords.LAT,
            example_coords.GALACTOCENTRIC_ROTATION,
            False,
            data.rgcentric,  # TODO fixtures
        ),
        # TODO the other datasets
    ],
)
def test_cartesian_model(
    test_data, lon, lat, rotation, deg: bool, expected_data
):
    """Test `~trackstream.preprocess.rotated_frame.cartesian_model`."""
    # --------------
    # setup

    angle_unit = u.deg if deg else u.rad
    # value, key map in expected_data
    rev_names = {
        v: k for k, v in expected_data.representation_component_names.items()
    }

    # --------------
    # apply model

    r, lon, lat = rotated_frame.cartesian_model(
        test_data.cartesian, lon=lon, lat=lat, rotation=rotation, deg=deg
    )
    # longitude and latitude
    lon = np.mod(coord.Longitude(lon, unit=angle_unit), 180 * u.deg)
    lat = np.mod(coord.Longitude(lat, unit=angle_unit), 180 * u.deg)

    # --------------
    # Testing r, lon, lat
    expected_r = getattr(expected_data, rev_names["distance"])
    expected_lon = getattr(expected_data, rev_names["lon"])
    expected_lat = getattr(expected_data, rev_names["lat"])

    assert_quantity_allclose(r, expected_r, atol=1e-10 * expected_r.unit)
    assert_quantity_allclose(
        lon, np.mod(expected_lon, 180 * u.deg), atol=1e-10 * expected_lon.unit
    )
    assert_quantity_allclose(
        lat, np.mod(expected_lat, 180 * u.deg), atol=1e-10 * expected_lat.unit
    )


# /def


# -------------------------------------------------------------------


@pytest.mark.parametrize(
    "test_data,variables,scalar,expected_lat",
    [
        (
            data.icrs,  # TODO fixtures
            (
                example_coords.ICRS_ROTATION,
                example_coords.RA,
                example_coords.DEC,
            ),
            True,
            0,  # TODO fixtures
        ),
        (
            data.icrs,  # TODO fixtures
            (
                example_coords.ICRS_ROTATION,
                example_coords.RA,
                example_coords.DEC,
            ),
            False,
            data.ricrs.phi2,  # TODO fixtures
        ),
        (
            data.gcentric,  # TODO fixtures
            (
                example_coords.GALACTOCENTRIC_ROTATION,
                example_coords.LON,
                example_coords.LAT,
            ),
            True,
            0,  # TODO fixtures
        ),
        (
            data.gcentric,  # TODO fixtures
            (
                example_coords.GALACTOCENTRIC_ROTATION,
                example_coords.LON,
                example_coords.LAT,
            ),
            False,
            data.rgcentric.phi2,  # TODO fixtures
        ),
        # TODO the other datasets
    ],
)
def test_residual(test_data, variables, scalar, expected_lat):
    """Test `~trackstream.preprocess.rotated_frame.residual`."""
    res = rotated_frame.residual(variables, test_data.cartesian, scalar=scalar)

    # compare result and expected latitude
    # When scalar is False, can have units, so always take value.
    np.testing.assert_allclose(
        res, u.Quantity(expected_lat).to_value(), atol=1e-12
    )


# /def


# -------------------------------------------------------------------


def test__make_bounds_defaults():
    """Test `~trackstream.preprocess.rotated_frame._make_bounds_defaults`."""
    expected = set(("rot_lower", "rot_upper", "origin_lim"))
    assert set(rotated_frame._make_bounds_defaults.keys()) == expected


# /def


# TODO use hypothesis instead
@pytest.mark.parametrize(
    "origin,rot_lower,rot_upper,origin_lim,expected",
    [
        (
            coord.UnitSphericalRepresentation(lon=0 * u.deg, lat=0 * u.deg),
            -1 * u.deg,
            2 * u.deg,
            3 * u.deg,
            np.c_[[-1, 2], [-3, 3], [-3, 3]].T,
        ),
        (
            coord.UnitSphericalRepresentation(lon=10 * u.deg, lat=11 * u.deg),
            -1 * u.deg,
            2 * u.deg,
            3 * u.deg,
            np.c_[[-1, 2], [7, 13], [8, 14]].T,
        ),
    ],
)
def test_make_bounds(
    origin: coord.UnitSphericalRepresentation,
    rot_lower: u.Quantity,
    rot_upper: u.Quantity,
    origin_lim: u.Quantity,
    expected: coord.UnitSphericalRepresentation,
):
    """Test `~trackstream.preprocess.rotated_frame.make_bounds`."""
    bounds = rotated_frame.make_bounds(
        origin, rot_lower=rot_lower, rot_upper=rot_upper, origin_lim=origin_lim
    )

    np.testing.assert_allclose(bounds, expected)


# /def


# -------------------------------------------------------------------


def test__minimize_defaults():
    """Test `~trackstream.preprocess.rotated_frame._minimize_defaults`."""
    expected = set(("fix_origin", "use_lmfit", "leastsquares", "align_v"))
    assert set(rotated_frame._minimize_defaults.keys()) == expected


# /def


@pytest.mark.skip(reason="TODO")
class Test_fit_frame:
    """Test `~trackstream.preprocess.rotated_frame.fit_frame`."""

    assert False


# /def


# -------------------------------------------------------------------


@pytest.mark.skip(reason="TODO")
def test_align_v_positive_lon():
    """Test `~trackstream.preprocess.rotated_frame.align_v_positive_lon`."""
    assert False


# /def


# -------------------------------------------------------------------


@pytest.mark.skip(reason="TODO")
def test_order_data_from_lon():
    """Test `~trackstream.preprocess.rotated_frame.order_data_from_lon`."""
    assert False


# /def


# -------------------------------------------------------------------


@pytest.mark.skip(reason="TODO")
class Test_RotatedFrameFitter:
    """Test `~trackstream.preprocess.rotated_frame.RotatedFrameFitter`."""

    def setup_class(self):
        """Setup testing class."""
        pass

    # /def

    def teardown_class(self):
        """Setup testing class."""
        pass

    # /def

    # -------------------------------------------

    @pytest.mark.skip(reason="TODO")
    def test_make_bounds(self):
        """Test method ``make_bounds``."""
        assert False

    # /def

    @pytest.mark.skip(reason="TODO")
    def test_fit(self):
        """Test method ``fit``."""
        assert False

    # /def

    @pytest.mark.skip(reason="TODO")
    def test_residual(self):
        """Test method ``residual``."""
        assert False

    # /def

    # -------------------------------------------
    # plot test methods

    @pytest.mark.skip(reason="TODO")
    @pytest.mark.mpl_image_compare(baseline_dir="baseline_images")
    def test_plot_data(self):
        """Test method ``plot_data``."""
        assert False

    # /def

    @pytest.mark.skip(reason="TODO")
    @pytest.mark.mpl_image_compare(baseline_dir="baseline_images")
    def test_plot_residual(self):
        """Test method ``plot_residual``."""
        assert False

    # /def


# /def

# -------------------------------------------------------------------


@pytest.mark.skip(reason="TODO")
class Test_FitResult:
    """Test `~trackstream.preprocess.rotated_frame.RotatedFrameFitter`."""

    def setup_class(self):
        """Setup testing class."""
        pass

    # /def

    def teardown_class(self):
        """Setup testing class."""
        pass

    # /def

    # -------------------------------------------

    @pytest.mark.skip(reason="TODO")
    def test_fit_values(self):
        """Test method ``fit_values``."""
        assert False

    # /def

    @pytest.mark.skip(reason="TODO")
    def test_frame(self):
        """Test method ``frame``."""
        assert False

    # /def

    @pytest.mark.skip(reason="TODO")
    def test_residual(self):
        """Test method ``residual``."""
        assert False

    # /def

    @pytest.mark.skip(reason="TODO")
    def test_residual_scalar(self):
        """Test method ``residual_scalar``."""
        assert False

    # /def

    @pytest.mark.skip(reason="TODO")
    def test_lon_order(self):
        """Test method ``residual_scalar``."""
        assert False

    # /def

    # -------------------------------------------
    # test plot methods

    @pytest.mark.skip(reason="TODO")
    @pytest.mark.mpl_image_compare(baseline_dir="baseline_images")
    def test_plot_data(self):
        """Test method ``plot_data``."""
        assert False

    # /def

    @pytest.mark.skip(reason="TODO")
    @pytest.mark.mpl_image_compare(baseline_dir="baseline_images")
    def test_plot_on_residual(self):
        """Test method ``plot_on_residual``."""
        assert False

    # /def


# /def


##############################################################################
# END
