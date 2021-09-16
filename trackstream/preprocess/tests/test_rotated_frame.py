# -*- coding: utf-8 -*-

"""Testing :mod:`~trackstream.preprocess.rotated_frame`.

.. todo::

    properly use pytest fixtures

"""

__all__ = [
    "test_cartesian_model",
    "test_residual",
    # "test__make_bounds_defaults",
    # "test_make_bounds",
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
    test_data,
    lon,
    lat,
    rotation,
    deg: bool,
    expected_data,
):
    """Test `~trackstream.preprocess.rotated_frame.cartesian_model`."""
    # --------------
    # setup

    angle_unit = u.deg if deg else u.rad  # get unit
    # reverse map: value, key map in expected_data
    rev_names = {v: k for k, v in expected_data.representation_component_names.items()}

    # --------------
    # apply model

    r, lon, lat = rotated_frame.cartesian_model(
        test_data.cartesian,
        lon=lon,
        lat=lat,
        rotation=rotation,
        deg=deg,
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
        lon,
        np.mod(expected_lon, 180 * u.deg),
        atol=1e-10 * expected_lon.unit,
    )
    assert_quantity_allclose(
        lat,
        np.mod(expected_lat, 180 * u.deg),
        atol=1e-10 * expected_lat.unit,
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
        res,
        u.Quantity(expected_lat).to_value(),
        atol=1e-12,
    )


# /def


##############################################################################
# END
