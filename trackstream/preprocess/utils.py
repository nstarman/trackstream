# -*- coding: utf-8 -*-
# see LICENSE.rst

"""Utilities.

This sub-module is destined for common non-package specific utility functions.

"""


__all__ = [
    # functions
    "cartesian_to_spherical",
    "reference_to_skyoffset_matrix",
]


##############################################################################
# IMPORTS

# BUILT-IN
import typing as T

# THIRD PARTY
import astropy.units as u
import numpy as np
from astropy.coordinates.matrix_utilities import (
    matrix_product,
    rotation_matrix,
)

# PROJECT-SPECIFIC
from trackstream.type_hints import CoordinateType, RepresentationType

##############################################################################
# PARAMETERS

DataType = T.Union[CoordinateType, RepresentationType]


##############################################################################
# CODE
##############################################################################


def cartesian_to_spherical(
    x: T.Sequence,
    y: T.Sequence,
    z: T.Sequence,
    deg: bool = False,
) -> T.Tuple[T.Sequence, T.Sequence, T.Sequence]:
    """Cartesian to Spherical.

    Adopts the Astropy ranges for `lon` and `lat`.

    Parameters
    ----------
    x, y, z : Sequence
    deg : bool
        Whether to return in degrees or radians -- default radians.

    Returns
    -------
    r, lat, lon : Sequence

    """
    r: T.Sequence = np.sqrt(x ** 2.0 + y ** 2.0 + z ** 2.0)
    lat: T.Sequence = (
        np.arctan2(np.sqrt(x ** 2.0 + y ** 2.0), z) - np.pi / 2
    )  # to match astropy
    lon: T.Sequence = np.arctan2(y, x) + np.pi  # to match astropy

    if deg:
        lat *= 180.0 / np.pi
        lon *= 180.0 / np.pi

    return r, lat, lon


# /def


# -------------------------------------------------------------------


def reference_to_skyoffset_matrix(
    lon: T.Union[float, u.Quantity],
    lat: T.Union[float, u.Quantity],
    rotation: T.Union[float, u.Quantity],
) -> T.Sequence:
    """Convert a reference coordinate to an sky offset frame [astropy].

    Cartesian to Cartesian matrix transform.

    .. |AngleType| replace:: :class:`~astropy.coordinates.Angle`
    .. |Quantity| replace:: :class:`~astropy.units.Quantity`

    Parameters
    ----------
    lon : float or |AngleType| or |Quantity| instance
        For ICRS, the right ascension.
        If float, assumed degrees.
    lat : |AngleType| or |Quantity| instance
        For ICRS, the declination.
        If float, assumed degrees.
    rotation : |AngleType| or |Quantity| instance
        The final rotation of the frame about the ``origin``. The sign of
        the rotation is the left-hand rule.  That is, an object at a
        particular position angle in the un-rotated system will be sent to
        the positive latitude (z) direction in the final frame.
        If float, assumed degrees.

    Returns
    -------
    ndarray
        (3x3) matrix. rotates reference Cartesian to skyoffset Cartesian.

    See Also
    --------
    :func:`~astropy.coordinates.builtin.skyoffset.reference_to_skyoffset`

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
    # Define rotation matrices along the position angle vector, and
    # relative to the origin.
    mat1 = rotation_matrix(-rotation, "x")
    mat2 = rotation_matrix(-lat, "y")
    mat3 = rotation_matrix(lon, "z")

    M = matrix_product(mat1, mat2, mat3)

    return M


# /def


##############################################################################
# END
