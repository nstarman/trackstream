# -*- coding: utf-8 -*-
# see LICENSE.rst

"""Initialization file.

This sub-module is destined for common non-package specific utility functions.

"""

__author__ = "Nathaniel Starkman"

__all__ = [
    "RotatedICRS",
    "icrs_to_rotated",
    "rotated_to_icrs",
]


##############################################################################
# IMPORTS

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
from astropy.coordinates.matrix_utilities import matrix_transpose
from astropy.utils.decorators import format_doc

# LOCAL
from trackstream.utils import reference_to_skyoffset_matrix

##############################################################################
# CODE
##############################################################################


class RotatedFrame(coord.BaseCoordinateFrame):
    """Example Rotated frame.

    Implemented from an Astropy [astropy]_ SkyOffset Frame.

    Parameters
    ----------
    representation : `~astropy.coordinates.BaseRepresentation` or None
        A representation object or None to have no data
        (or use the other keywords)
    phi1 : `~astropy.coordinates.Angle`, optional, must be keyword
        The longitude-like angle corresponding to Sagittarius' orbit.
    phi2 : `~astropy.coordinates.Angle`, optional, must be keyword
        The latitude-like angle corresponding to Sagittarius' orbit.
    distance : `Quantity`, optional, must be keyword
        The Distance for this object along the line-of-sight.
    pm_phi1_cosphi2 : :class:`~astropy.units.Quantity`, optional, keyword
        The proper motion along the stream in ``Lambda`` (including the
        ``cos(Beta)`` factor) for this object (``pm_Beta`` must also be given).
    pm_phi2 : :class:`~astropy.units.Quantity`, optional, must be keyword
        The proper motion in Declination for this object (``pm_ra_cosdec`` must
        also be given).
    radial_velocity : :class:`~astropy.units.Quantity`, optional, keyword
        The radial velocity of this object.

    References
    ----------
    .. [astropy] Astropy Collaboration, Robitaille, T., Tollerud, E.,
        Greenfield, P., Droettboom, M., Bray, E., Aldcroft, T., Davis,
        M., Ginsburg, A., Price-Whelan, A., Kerzendorf, W., Conley, A.,
        Crighton, N., Barbary, K., Muna, D., Ferguson, H., Grollier, F.,
        Parikh, M., Nair, P., Unther, H., Deil, C., Woillez, J.,
        Conseil, S., Kramer, R., Turner, J., Singer, L., Fox, R.,
        Weaver, B., Zabalza, V., Edwards, Z., Azalee Bostroem, K.,
        Burke, D., Casey, A., Crawford, S., Dencheva, N., Ely, J.,
        Jenness, T., Labrie, K., Lim, P., Pierfederici, F., Pontzen, A.,
        Ptak, A., Refsdal, B., Servillat, M., & Streicher, O. (2013).
        Astropy: A community Python package for astronomy.
        Astronomy and Astrophysics, 558, A33.

    """

    default_representation = coord.SphericalRepresentation
    default_differential = coord.SphericalCosLatDifferential

    frame_specific_representation_info = {
        coord.SphericalRepresentation: [
            coord.RepresentationMapping("lon", "phi1"),
            coord.RepresentationMapping("lat", "phi2"),
            coord.RepresentationMapping("distance", "distance"),
        ],
    }


# /class


# ------------------------------------------------------------------


@format_doc(RotatedFrame.__doc__)
class RotatedICRS(RotatedFrame):
    """Example Rotated ICRS frame."""

    pass


# /class


# Generate the rotation matrix
RA = 20 * u.deg
DEC = 30 * u.deg
ICRS_ROTATION = 135.7 * u.deg
ICRS_ROT_MATRIX = reference_to_skyoffset_matrix(RA, DEC, ICRS_ROTATION)


@coord.frame_transform_graph.transform(
    coord.StaticMatrixTransform,
    coord.ICRS,
    RotatedICRS,
)
def icrs_to_rotated():
    """Transformation matrix from ICRS Cartesian to rotated coordinates."""
    return ICRS_ROT_MATRIX


@coord.frame_transform_graph.transform(
    coord.StaticMatrixTransform,
    RotatedICRS,
    coord.ICRS,
)
def rotated_to_icrs():
    """Transformation matrix from rotated coordinates to ICRS Cartesian."""
    return matrix_transpose(ICRS_ROT_MATRIX)


# ------------------------------------------------------------------


@format_doc(RotatedFrame.__doc__)
class RotatedGalactocentric(RotatedFrame):
    """Example Rotated Galactocentric frame."""

    pass


# /class


# Generate the rotation matrix
LON = 20 * u.deg
LAT = 30 * u.deg
GALACTOCENTRIC_ROTATION = 135.7 * u.deg
GALACTOCENTRIC_ROT_MATRIX = reference_to_skyoffset_matrix(
    LON,
    LAT,
    GALACTOCENTRIC_ROTATION,
)


@coord.frame_transform_graph.transform(
    coord.StaticMatrixTransform,
    coord.Galactocentric,
    RotatedGalactocentric,
)
def Galactocentric_to_rotated():
    """Transformation matrix from GC Cartesian to rotated coordinates."""
    return GALACTOCENTRIC_ROT_MATRIX


@coord.frame_transform_graph.transform(
    coord.StaticMatrixTransform,
    RotatedGalactocentric,
    coord.Galactocentric,
)
def rotated_to_Galactocentric():
    """Transformation matrix from rotated coordinates to GC Cartesian."""
    return matrix_transpose(GALACTOCENTRIC_ROT_MATRIX)


##############################################################################
# END
