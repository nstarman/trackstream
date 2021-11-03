# -*- coding: utf-8 -*-

"""Coordinates Utilities."""


__all__ = ["cartesian_to_spherical", "reference_to_skyoffset_matrix", "resolve_framelike"]


##############################################################################
# IMPORTS

# STDLIB
import functools
import typing as T

# THIRD PARTY
import astropy.units as u
import numpy as np
from astropy.coordinates import BaseCoordinateFrame, SkyCoord, sky_coordinate_parsers
from astropy.coordinates.matrix_utilities import matrix_product, rotation_matrix

# LOCAL
from trackstream._type_hints import ArrayLike
from trackstream.config import conf

##############################################################################
# PARAMETERS


##############################################################################
# CODE
##############################################################################


def cartesian_to_spherical(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    deg: bool = False,
) -> T.Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Cartesian to Spherical.

    Adopts the Astropy ranges for `lon` and `lat`.

    Parameters
    ----------
    x, y, z : scalar or ndarray
    deg : bool
        Whether to return in degrees or radians -- default radians.

    Returns
    -------
    r, lat, lon : scalar or ndarray
    """
    r = np.sqrt(x ** 2.0 + y ** 2.0 + z ** 2.0)
    lat = np.arctan2(np.sqrt(x ** 2.0 + y ** 2.0), z) - np.pi / 2  # to match astropy
    lon = np.arctan2(y, x) + np.pi  # to match astropy

    if deg:
        lat *= 180.0 / np.pi
        lon *= 180.0 / np.pi

    return lon, lat, r


def reference_to_skyoffset_matrix(
    lon: T.Union[float, u.Quantity],
    lat: T.Union[float, u.Quantity],
    rotation: T.Union[float, u.Quantity],
) -> np.ndarray:
    """Convert a reference coordinate to an sky offset frame [astropy].

    Cartesian to Cartesian matrix transform.

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


@functools.singledispatch
def resolve_framelike(frame, error_if_not_type: bool = True):
    """Determine the frame and return a blank instance.

    Parameters
    ----------
    frame : frame-like instance or None (optional)
        If BaseCoordinateFrame, replicates without data.
        If str, uses astropy parsers to determine frame class
        If None (default), gets default frame name from config, and parses.

    error_if_not_type : bool
        Whether to raise TypeError if `frame` is not one of the allowed types.

    Returns
    -------
    frame : `~astropy.coordinates.BaseCoordinateFrame` instance
        Replicated without data.

    Raises
    ------
    TypeError
        If `frame` is not one of the allowed types and 'error_if_not_type' is
        True.
    """
    if error_if_not_type:
        raise TypeError(
            "Input coordinate frame must be an astropy "
            "coordinates frame subclass *instance*, not a "
            "'{}'".format(frame.__class__.__name__)
        )
    return frame


@resolve_framelike.register
def _(frame: None, error_if_not_type: bool = True) -> BaseCoordinateFrame:
    # If no frame is specified, assume that the input footprint is in a
    # frame specified in the configuration
    return resolve_framelike(conf.default_frame)


@resolve_framelike.register
def _(frame: str, error_if_not_type: bool = True) -> BaseCoordinateFrame:
    # strings can be turned into frames using the private SkyCoord parsers
    out: BaseCoordinateFrame = sky_coordinate_parsers._get_frame_class(frame.lower())()
    return out


@resolve_framelike.register
def _(frame: str, rror_if_not_type: bool = True) -> BaseCoordinateFrame:
    out: BaseCoordinateFrame = frame.replicate_without_data()
    return out


@resolve_framelike.register
def _(frame: SkyCoord, rror_if_not_type: bool = True) -> BaseCoordinateFrame:
    out: BaseCoordinateFrame = frame.frame.replicate_without_data()
    return out


##############################################################################
# END
