# -*- coding: utf-8 -*-

"""Coordinates Utilities."""


__all__ = ["reference_to_skyoffset_matrix", "resolve_framelike"]


##############################################################################
# IMPORTS

# STDLIB
import functools
from typing import Any, Optional, Tuple, Type, TypeVar, Union, cast, overload

# THIRD PARTY
from astropy.coordinates import BaseCoordinateFrame as BaseFrame
from astropy.coordinates import BaseDifferential as BaseDif
from astropy.coordinates import BaseRepresentation as BaseRep
from astropy.coordinates import SkyCoord
from astropy.coordinates.matrix_utilities import matrix_product, rotation_matrix
from astropy.coordinates.sky_coordinate_parsers import _get_frame_class
from astropy.units import Quantity
from numpy import arcsin, arctan2, broadcast_to, cos, ndarray, pi, sin

# LOCAL
from trackstream._type_hints import CoordinateType

##############################################################################
# PARAMETERS

_PI_2 = pi / 2

_FT = TypeVar("_FT", bound=BaseFrame)
_RT = TypeVar("_RT", bound=BaseRep)

##############################################################################
# CODE
##############################################################################


def reference_to_skyoffset_matrix(
    lon: Union[float, Quantity],
    lat: Union[float, Quantity],
    rotation: Union[float, Quantity],
) -> ndarray:
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

    M: ndarray = matrix_product(mat1, mat2, mat3)

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
def resolve_framelike(frame: SkyCoord, type_error: bool = True) -> BaseFrame:  # type: ignore  # noqa: E501, F811
    out: BaseFrame = frame.frame.replicate_without_data()
    out.representation_type = frame.representation_type
    return out


def resolve_framelike(  # type: ignore
    frame: Union[str, BaseFrame, SkyCoord],
    type_error: bool = True,
) -> BaseFrame:  # noqa: F811
    """Determine the frame and return a blank instance.

    Parameters
    ----------
    frame : frame-like instance or None (optional)
        If BaseCoordianteFrame, replicates without data.
        If str, uses astropy parsers to determine frame class

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


# -------------------------------------------------------------------


@overload
def deep_transform_to(
    crd: SkyCoord,
    frame: _FT,
    representation_type: Type[BaseRep],
    differential_type: Optional[type[BaseDif]] = None,
) -> SkyCoord:
    ...


@overload
def deep_transform_to(
    crd: BaseFrame,
    frame: _FT,
    representation_type: Type[BaseRep],
    differential_type: Optional[type[BaseDif]] = None,
) -> _FT:
    ...


def deep_transform_to(
    crd: Union[SkyCoord, BaseFrame],
    frame: _FT,
    representation_type: Type[_RT],
    differential_type: Optional[type[BaseDif]] = None,
) -> Union[SkyCoord, _FT]:
    """Transform a coordinate to a frame and representation type.

    For speed, Astropy transformations can be shallow. This function does
    ``.transform_to(frame, representation_type=representation_type)`` and makes
    sure all the underlying data is actually in the desired representation type.

    Parameters
    ----------
    crd : SkyCoord or BaseCoordinateFrame
    frame : BaseCoordinateFrame
    representation_type : BaseRepresentation class

    Returns
    -------
    crd : SkyCoord or BaseCoordinateFrame
        Transformed to ``frame`` and ``representation_type``.
    """
    c: CoordinateType = crd.transform_to(frame)
    r: _RT = c.represent_as(representation_type)
    # TODO! differential_type

    data = cast(
        _FT,
        c.realize_frame(
            r,
            representation_type=representation_type,
            differential_type=differential_type,
            copy=False,
        ),
    )
    out = SkyCoord(data, copy=False) if isinstance(crd, SkyCoord) else data

    return out


# -------------------------------------------------------------------
# Copied from Astropy, with units stripped. See License
# https://docs.astropy.org/en/stable/_modules/astropy/coordinates/angle_utilities.html
# The implementations in Astropy use Angle and are too slow.


def position_angle(lon1: ndarray, lat1: ndarray, lon2: float, lat2: float) -> ndarray:
    """
    Position Angle (East of North) between two points on a sphere.

    Parameters
    ----------
    lon1, lat1, lon2, lat2 : float['radian']
        Longitude and latitude of the two points. Quantities should be in
        angular units; floats in radians.

    Returns
    -------
    pa : float['radian']
        The (positive) position angle of the vector pointing from position 1 to
        position 2.  If any of the angles are arrays, this will contain an array
        following the appropriate `numpy` broadcasting rules.
    """
    deltalon = lon2 - lon1
    colat = cos(lat2)

    x = sin(lat2) * cos(lat1) - colat * sin(lat1) * cos(deltalon)
    y = sin(deltalon) * colat

    return arctan2(y, x)


def offset_by(
    lon: ndarray, lat: ndarray, posang: ndarray, distance: ndarray
) -> Tuple[ndarray, ndarray]:
    """
    Point with the given offset from the given point.

    Parameters
    ----------
    lon, lat, posang, distance : float['rad']
        Longitude and latitude of the starting point,
        position angle and distance to the final point.
        Polar points at lat= +/-90 are treated as limit of +/-(90-epsilon) and same lon.

    Returns
    -------
    lon, lat : float['rad']
        The position of the final point.  If any of the angles are arrays,
        these will contain arrays following the appropriate `numpy` broadcasting rules.
        0 <= lon < 2pi.
    """
    # Calculations are done using the spherical trigonometry sine and cosine rules
    # of the triangle A at North Pole,   B at starting point,   C at final point
    # with angles  A (change in lon), B (posang),     C (not used, but negative reciprocal posang)
    # with sides      a (distance),      b (final co-latitude), c (starting colatitude)
    # B, a, c are knowns; A and b are unknowns
    # https://en.wikipedia.org/wiki/Spherical_trigonometry

    cos_a = cos(distance)
    sin_a = sin(distance)
    cos_c = sin(lat)
    sin_c = cos(lat)
    cos_B = cos(posang)
    sin_B = sin(posang)

    # cosine rule: Know two sides: a,c and included angle: B; get unknown side b
    cos_b = cos_c * cos_a + sin_c * sin_a * cos_B
    # sin_b = sqrt(1 - cos_b**2)
    # sine rule and cosine rule for A (using both lets arctan2 pick quadrant).
    # multiplying both sin_A and cos_A by x=sin_b * sin_c prevents /0 errors
    # at poles.  Correct for the x=0 multiplication a few lines down.
    # sin_A/sin_a == sin_B/sin_b    # Sine rule
    xsin_A = sin_a * sin_B * sin_c
    # cos_a == cos_b * cos_c + sin_b * sin_c * cos_A  # cosine rule
    xcos_A = cos_a - cos_b * cos_c  # type: ignore

    A = arctan2(xsin_A, xcos_A)  # radian
    # Treat the poles as if they are infinitesimally far from pole but at given lon
    small_sin_c = sin_c < 1e-12
    if small_sin_c.any():
        # For south pole (cos_c = -1), A = posang; for North pole, A=180 deg - posang
        A_pole = _PI_2 + cos_c * (_PI_2 - posang)
        if A.shape:
            # broadcast to ensure the shape is like that of A, which is also
            # affected by the (possible) shapes of lat, posang, and distance.
            small_sin_c = broadcast_to(small_sin_c, A.shape)
            A[small_sin_c] = A_pole[small_sin_c]
        else:
            A = A_pole

    outlon = lon + A
    outlat = arcsin(cos_b)

    return outlon, outlat
