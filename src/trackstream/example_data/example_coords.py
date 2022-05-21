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
import astropy.units as u
from astropy.coordinates import ICRS, BaseCoordinateFrame, Galactocentric, RepresentationMapping
from astropy.coordinates import SphericalCosLatDifferential, SphericalRepresentation
from astropy.coordinates import StaticMatrixTransform, frame_transform_graph
from astropy.coordinates.matrix_utilities import matrix_transpose
from astropy.units import Quantity
from astropy.utils.decorators import format_doc
from numpy import ndarray

# LOCAL
from trackstream.utils import reference_to_skyoffset_matrix

##############################################################################
# CODE
##############################################################################


class RotatedFrame(BaseCoordinateFrame):
    """Example Rotated frame.

    Implemented from an Astropy [astropy]_ SkyOffset Frame.

    Parameters
    ----------
    representation : `~astropy.coordinates.BaseRepresentation` or None
        A representation object or None to have no data
        (or use the other keywords)
    phi1 : `~astropy.coordinates.Angle`, optional, must be keyword
        The |longitude|-like angle corresponding to Sagittarius' orbit.
    phi2 : `~astropy.coordinates.Angle`, optional, must be keyword
        The |Latitude|-like angle corresponding to Sagittarius' orbit.
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

    default_representation = SphericalRepresentation
    default_differential = SphericalCosLatDifferential

    frame_specific_representation_info = {
        SphericalRepresentation: [
            RepresentationMapping("lon", "phi1"),
            RepresentationMapping("lat", "phi2"),
            RepresentationMapping("distance", "distance"),
        ],
    }


# ------------------------------------------------------------------


@format_doc(RotatedFrame.__doc__)
class RotatedICRS(RotatedFrame):
    """Example Rotated ICRS frame."""


# Generate the rotation matrix
RA = Quantity(20, u.deg)
DEC = Quantity(30, u.deg)
ICRS_ROTATION = Quantity(135.7, u.deg)
ICRS_ROT_MATRIX = reference_to_skyoffset_matrix(RA, DEC, ICRS_ROTATION)


@frame_transform_graph.transform(
    StaticMatrixTransform,
    ICRS,
    RotatedICRS,
)
def icrs_to_rotated() -> ndarray:
    """Transformation matrix from ICRS Cartesian to rotated coordinates."""
    return ICRS_ROT_MATRIX


@frame_transform_graph.transform(
    StaticMatrixTransform,
    RotatedICRS,
    ICRS,
)
def rotated_to_icrs() -> ndarray:
    """Transformation matrix from rotated coordinates to ICRS Cartesian."""
    matrix: ndarray = matrix_transpose(ICRS_ROT_MATRIX)
    return matrix


# ------------------------------------------------------------------


@format_doc(RotatedFrame.__doc__)
class RotatedGalactocentric(RotatedFrame):
    """Example Rotated Galactocentric frame."""


# Generate the rotation matrix
LON = Quantity(20, u.deg)
LAT = Quantity(30, u.deg)
GALACTOCENTRIC_ROTATION = Quantity(135.7, u.deg)
GALACTOCENTRIC_ROT_MATRIX = reference_to_skyoffset_matrix(
    LON,
    LAT,
    GALACTOCENTRIC_ROTATION,
)


@frame_transform_graph.transform(
    StaticMatrixTransform,
    Galactocentric,
    RotatedGalactocentric,
)
def Galactocentric_to_rotated() -> ndarray:
    """Transformation matrix from GC Cartesian to rotated coordinates."""
    return GALACTOCENTRIC_ROT_MATRIX


@frame_transform_graph.transform(
    StaticMatrixTransform,
    RotatedGalactocentric,
    Galactocentric,
)
def rotated_to_Galactocentric() -> ndarray:
    """Transformation matrix from rotated coordinates to GC Cartesian."""
    matrix: ndarray = matrix_transpose(GALACTOCENTRIC_ROT_MATRIX)
    return matrix
