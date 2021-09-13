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


###############################################################################
# IMPORTS

# BUILT-IN
import typing as T
from collections.abc import Sequence

# THIRD PARTY
import astropy.coordinates as coord
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
    # TODO! re-implement with pyerfa for speed
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

# -------------------------------------------------------------------


def find_closest_point(
    data: DataType,
    near_point: T.Union[RepresentationType, T.Sequence, np.ndarray],
):
    """Find starting point.

    .. |Rep| replace:: :class:`~astropy.coordinates.BaseRepresentation`
    .. |Coord| replace:: :class:`~astropy.coordinates.BaseCoordinateFrame`

    Parameters
    ----------
    data : |Rep| or |Coord| instance
        Shape (# measurements, # features).
        Must be transformable to Cartesian coordinates.
    near_point : Sequence
        Shape (1, # features)
        If passing an array, can reshape with ``.reshape(1, -1)``

    Returns
    -------
    start_point : Sequence
        Shape (# features, ). Point in `data` nearest `near_point` in KDTree
    start_ind : int
        Index into `data` for the `start_point`
        If `return_ind` == True

    """
    if isinstance(near_point, (Sequence, np.ndarray)):
        near_point = coord.CartesianRepresentation(near_point)
    else:
        near_point = near_point.represent_as(coord.CartesianRepresentation)

    data = data.represent_as(coord.CartesianRepresentation)

    start_ind = np.argmin((data - near_point).norm())
    start_point = data[start_ind]

    return start_point, start_ind


# /def


# -------------------------------------------------------------------


def set_starting_point(data: DataType, start_ind: int):
    """Reorder data to set starting index at row 0.

    Parameters
    ----------
    data
    start_ind

    Returns
    -------
    `data`
        re-ordered

    """
    # index order array
    order = list(range(len(data)))
    del order[start_ind]
    order = np.array([start_ind, *order])

    return data[order]  # return reordered data


# /def


# -------------------------------------------------------------------


# def select_bind_partial(sig, *args, **kwargs):
#     """Get a BoundArguments object, that partially maps the
#     passed `args` and `kwargs` to the function's signature.
#     Raises `TypeError` if the passed arguments can not be bound.
#     """
#     return _select_bind(sig, args, kwargs, partial=True)
#
#
# # /def
#
#
# def _select_bind(sig, args, kwargs, *, partial=False):
#     """Bind partial, not throwing error if kwargs mismatch."""
#
#     arguments = dict()
#
#     parameters = iter(sig.parameters.values())
#     parameters_ex = ()
#     arg_vals = iter(args)
#
#     while True:
#         # Let's iterate through the positional arguments and corresponding
#         # parameters
#         try:
#             arg_val = next(arg_vals)
#         except StopIteration:
#             # No more positional arguments
#             try:
#                 param = next(parameters)
#             except StopIteration:
#                 # No more parameters. That's it. Just need to check that
#                 # we have no `kwargs` after this while loop
#                 break
#             else:
#                 if param.kind == _VAR_POSITIONAL:
#                     # That's OK, just empty *args.  Let's start parsing
#                     # kwargs
#                     break
#                 elif param.name in kwargs:
#                     if param.kind == _POSITIONAL_ONLY:
#                         msg = (
#                             "{arg!r} parameter is positional only, "
#                             "but was passed as a keyword"
#                         )
#                         msg = msg.format(arg=param.name)
#                         raise TypeError(msg) from None
#                     parameters_ex = (param,)
#                     break
#                 elif param.kind == _VAR_KEYWORD or param.default is not _empty:
#                     # That's fine too - we have a default value for this
#                     # parameter.  So, lets start parsing `kwargs`, starting
#                     # with the current parameter
#                     parameters_ex = (param,)
#                     break
#                 else:
#                     # No default, not VAR_KEYWORD, not VAR_POSITIONAL,
#                     # not in `kwargs`
#                     if partial:
#                         parameters_ex = (param,)
#                         break
#                     else:
#                         msg = "missing a required argument: {arg!r}"
#                         msg = msg.format(arg=param.name)
#                         raise TypeError(msg) from None
#         else:
#             # We have a positional argument to process
#             try:
#                 param = next(parameters)
#             except StopIteration:
#                 raise TypeError("too many positional arguments") from None
#             else:
#                 if param.kind in (_VAR_KEYWORD, _KEYWORD_ONLY):
#                     # Looks like we have no parameter for this positional
#                     # argument
#                     raise TypeError("too many positional arguments") from None
#
#                 if param.kind == _VAR_POSITIONAL:
#                     # We have an '*args'-like argument, let's fill it with
#                     # all positional arguments we have left and move on to
#                     # the next phase
#                     values = [arg_val]
#                     values.extend(arg_vals)
#                     arguments[param.name] = tuple(values)
#                     break
#
#                 if param.name in kwargs and param.kind != _POSITIONAL_ONLY:
#                     raise TypeError(
#                         "multiple values for argument {arg!r}".format(
#                             arg=param.name
#                         )
#                     ) from None
#
#                 arguments[param.name] = arg_val
#
#     # Now, we iterate through the remaining parameters to process
#     # keyword arguments
#     kwargs_param = None
#     for param in itertools.chain(parameters_ex, parameters):
#         if param.kind == _VAR_KEYWORD:
#             # Memorize that we have a '**kwargs'-like parameter
#             kwargs_param = param
#             continue
#
#         if param.kind == _VAR_POSITIONAL:
#             # Named arguments don't refer to '*args'-like parameters.
#             # We only arrive here if the positional arguments ended
#             # before reaching the last parameter before *args.
#             continue
#
#         param_name = param.name
#         try:
#             arg_val = kwargs.pop(param_name)
#         except KeyError:
#             # We have no value for this parameter.  It's fine though,
#             # if it has a default value, or it is an '*args'-like
#             # parameter, left alone by the processing of positional
#             # arguments.
#             if (
#                 not partial
#                 and param.kind != _VAR_POSITIONAL
#                 and param.default is _empty
#             ):
#                 raise TypeError(
#                     "missing a required argument: {arg!r}".format(
#                         arg=param_name
#                     )
#                 ) from None
#
#         else:
#             if param.kind == _POSITIONAL_ONLY:
#                 # This should never happen in case of a properly built
#                 # Signature object (but let's have this check here
#                 # to ensure correct behaviour just in case)
#                 raise TypeError(
#                     "{arg!r} parameter is positional only, "
#                     "but was passed as a keyword".format(arg=param.name)
#                 )
#
#             arguments[param_name] = arg_val
#
#     if kwargs:
#         if kwargs_param is not None:
#             # Process our '**kwargs'-like parameter
#             arguments[kwargs_param.name] = kwargs
#
#     return sig._bound_arguments_cls(sig, arguments)
#
#
# # /def


##############################################################################
# END
