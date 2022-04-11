# -*- coding: utf-8 -*-

"""Coordinates Utilities."""


__all__ = ["reference_to_skyoffset_matrix", "resolve_framelike"]


##############################################################################
# IMPORTS

# STDLIB
import functools
from typing import Any, Type, Union, cast, overload

# THIRD PARTY
import astropy.units as u
import numpy as np
from astropy.coordinates import BaseCoordinateFrame as BaseFrame
from astropy.coordinates import SkyCoord
from astropy.coordinates.sky_coordinate_parsers import _get_frame_class
from astropy.coordinates.matrix_utilities import matrix_product, rotation_matrix

##############################################################################
# CODE
##############################################################################


def reference_to_skyoffset_matrix(
    lon: Union[float, u.Quantity],
    lat: Union[float, u.Quantity],
    rotation: Union[float, u.Quantity],
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

    M: np.ndarray = matrix_product(mat1, mat2, mat3)

    return M


# -------------------------------------------------------------------
# For implementation explanation, see
# https://github.com/python/mypy/issues/8356#issuecomment-884548381


@functools.singledispatch
def _resolve_framelike(frame: Any, type_error: bool = True) -> BaseFrame:
    if type_error:
        raise TypeError(
            "Input coordinate frame must be an astropy "
            "coordinates frame subclass *instance*, not a "
            "'{}'".format(frame.__class__.__name__),
        )
    return frame


@overload
@_resolve_framelike.register
def resolve_framelike(frame: str, type_error: bool = True) -> BaseFrame:  # noqa: F811
    # strings can be turned into frames using the private SkyCoord parsers
    out: BaseFrame = cast(Type[BaseFrame], _get_frame_class(frame.lower()))()
    return out


@overload
@_resolve_framelike.register
def resolve_framelike(frame: BaseFrame, type_error: bool = True) -> BaseFrame:
    out: BaseFrame = frame.replicate_without_data()
    out.representation_type = frame.representation_type
    return out


@overload
@_resolve_framelike.register
def resolve_framelike(frame: SkyCoord, type_error: bool = True) -> BaseFrame:  # noqa: E501, F811
    out: BaseFrame = frame.frame.replicate_without_data()
    out.representation_type = frame.representation_type
    return out


def resolve_framelike(  # type: ignore
    frame: Union[str, BaseFrame, SkyCoord], type_error: bool = True
) -> BaseFrame:  # noqa: F811
    """Determine the frame and return a blank instance.

    Parameters
    ----------
    frame : frame-like instance or None (optional)
        If BaseCoordianteFrame, replicates without data.
        If str, uses astropy parsers to determine frame class
        If None (default), gets default frame name from config, and parses.

    type_error : bool
        Whether to raise TypeError if `frame` is not one of the allowed types.

    Returns
    -------
    frame : `~astropy.coordinates.BaseCoordianteFrame` instance
        Replicated without data.

    Raises
    ------
    TypeError
        If `frame` is not one of the allowed types and 'type_error' is
        True.
    """
    return _resolve_framelike(frame, type_error=type_error)
