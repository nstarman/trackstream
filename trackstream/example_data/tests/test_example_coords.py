# -*- coding: utf-8 -*-

"""Testing :mod:`~trackstream.example_data.example_coords`.

.. todo::

    properly use pytest fixtures

"""

__all__ = [
    "test_transformation_machinery",
]


##############################################################################
# IMPORTS

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import pytest
from astropy.tests.helper import assert_quantity_allclose

# PROJECT-SPECIFIC
from trackstream.example_data import example_coords

##############################################################################
# Fixtures


@pytest.fixture
def icrs():
    """ICRS data from ``trackstream.example_data.tests.data``.

    .. todo::

        properly use pytest fixtures

    """
    # PROJECT-SPECIFIC
    from trackstream.example_data.tests import data

    return data.icrs


# /def


@pytest.fixture
def ricrs():
    """Rotated ICRS data from ``trackstream.example_data.tests.data``.

    .. todo::

        properly use pytest fixtures

    """
    # PROJECT-SPECIFIC
    from trackstream.example_data.tests import data

    return data.ricrs


# /def


##############################################################################
# TESTS
##############################################################################


# @pytest.mark.mpl_image_compare(baseline_dir="baseline_images")
# def test_rotation_plot(icrs, ricrs):
#     """Test That the ICRS and rotated ICRS appear as they should."""
#     import matplotlib
#     import matplotlib.pyplot as pyplot

#     matplotlib.use("Agg")  # using a backend that doesn't display to the user

#     fig, axes = pyplot.subplots(
#         1, 2, figsize=(10, 5), subplot_kw={"projection": "aitoff"}
#     )

#     axes[0].set_title("Rotated ICRS")
#     axes[0].plot(
#         ricrs.phi1.wrap_at(180 * u.deg).radian,
#         ricrs.phi2.radian,
#         linestyle="none",
#         marker=".",
#     )

#     axes[1].set_title("ICRS")
#     axes[1].plot(
#         icrs.ra.wrap_at(180 * u.deg).radian,
#         icrs.dec.radian,
#         linestyle="none",
#         marker=".",
#     )

#     pyplot.tight_layout()

#     return fig


# # /def


# -------------------------------------------------------------------


def test_transformation_machinery(icrs, ricrs):
    """Test a round-trip conversion."""
    # convert from ICRS to rotated

    new_ricrs = icrs.transform_to(example_coords.RotatedICRS)

    assert_quantity_allclose(
        new_ricrs.phi1,
        ricrs.phi1,
        rtol=1e-10,
        atol=1e-7 * u.deg,
    )
    assert_quantity_allclose(
        new_ricrs.phi2,
        ricrs.phi2,
        rtol=1e-10,
        atol=1e-7 * u.deg,
    )
    assert_quantity_allclose(
        new_ricrs.distance,
        ricrs.distance,
        rtol=1e-10,
        atol=1e-7,
    )

    # and check it's close to 0

    assert_quantity_allclose(
        new_ricrs.phi2,
        0 * u.deg,
        rtol=1e-10,
        atol=1e-7 * u.deg,
    )

    # and back again

    new_icrs = new_ricrs.transform_to(coord.ICRS)

    assert_quantity_allclose(
        new_icrs.ra,
        icrs.ra,
        rtol=1e-10,
        atol=1e-7 * u.deg,
    )
    assert_quantity_allclose(
        new_icrs.dec,
        icrs.dec,
        rtol=1e-10,
        atol=1e-7 * u.deg,
    )
    assert_quantity_allclose(
        new_icrs.distance,
        icrs.distance,
        rtol=1e-10,
        atol=1e-7,
    )


# /def


# -------------------------------------------------------------------


##############################################################################
# END
