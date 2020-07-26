# -*- coding: utf-8 -*-

"""Coordinates."""


__all__ = [
    # functions
    "cartesian_to_spherical",
    "reference_to_skyoffset_matrix",
    "get_transform_matrix",
]


##############################################################################
# IMPORTS


# BUILT-IN

import typing as T


# THIRD PARTY

import astropy.coordinates as coord
from astropy.coordinates.matrix_utilities import (
    matrix_product,
    rotation_matrix,
    matrix_transpose,
)

import numpy as np
from utilipy.utils.typing import array_like, AngleType, CoordinateType


# PROJECT-SPECIFIC


##############################################################################
# PARAMETERS


##############################################################################
# CODE
##############################################################################


def cartesian_to_spherical(
    x: array_like, y: array_like, z: array_like, deg: bool = False
) -> T.Tuple[array_like, array_like, array_like]:
    """Cartesian to Spherical.

    Adopts the Astropy ranges for `lon` and `lat`.

    Parameters
    ----------
    x, y, z : array_like
    deg : bool
        Whether to return in degrees or radians -- default radians.

    Returns
    -------
    r, lat, lon : array_like

    """
    r: array_like = np.sqrt(x ** 2.0 + y ** 2.0 + z ** 2.0)
    lat: array_like = (
        np.arctan2(np.sqrt(x ** 2.0 + y ** 2.0), z) - np.pi / 2
    )  # to match astropy
    lon: array_like = np.arctan2(y, x) + np.pi  # to match astropy

    if deg:
        lat *= 180.0 / np.pi
        lon *= 180.0 / np.pi

    return r, lat, lon


# /def


# -------------------------------------------------------------------


def reference_to_skyoffset_matrix(
    lon: T.Union[float, AngleType],
    lat: T.Union[float, AngleType],
    rotation: T.Union[float, AngleType],
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

    return matrix_transpose(M)


# /def


# -------------------------------------------------------------------


def get_transform_matrix(
    from_frame: CoordinateType, to_frame: CoordinateType
) -> T.Sequence:
    """Get Transformation Matrix from Astropy [astropy]_.

    Compose sequential matrix transformations (static or dynamic) to get a
    single transformation matrix from a given path through the Astropy
    transformation machinery.

    Function modified from [gala1]_, [gala2]_.

    Parameters
    ----------
    from_frame : `~astropy.coordinates.BaseCoordinateFrame` subclass
        The class or instance of the frame you're transforming from.
    to_frame : `~astropy.coordinates.BaseCoordinateFrame` subclass
        The class or instance of the frame you're transforming to.

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

    .. [gala1] Adrian Price-Whelan and Brigitta Sipocz and Syrtis Major
        and Semyeong Oh. (2017). adrn/gala: v0.2.1.

    .. [gala2] Adrian M. Price-Whelan (2017). Gala: A Python package
        for galactic dynamicsThe Journal of Open Source Software, 2(18).

    See Also
    --------
    :func:`~gala.coordinates.pm_cov_traansform.get_transform_matrix`
        The original function, of which this is a copy.

    """
    if isinstance(from_frame, coord.BaseCoordinateFrame):
        from_frame_cls = from_frame.__class__
    else:
        from_frame_cls = from_frame

    if isinstance(to_frame, coord.BaseCoordinateFrame):
        to_frame_cls = to_frame.__class__
    else:
        to_frame_cls = to_frame

    path, distance = coord.frame_transform_graph.find_shortest_path(
        from_frame_cls, to_frame_cls
    )

    matrices = []
    currsys = from_frame
    for p in path[1:]:  # first element is fromsys so we skip it
        if isinstance(currsys, coord.BaseCoordinateFrame):
            currsys_cls = currsys.__class__
        else:
            currsys_cls = currsys
            currsys = currsys_cls()

        trans = coord.frame_transform_graph._graph[currsys_cls][p]

        if isinstance(to_frame, p):
            p = to_frame

        if isinstance(trans, coord.DynamicMatrixTransform):
            M = trans.matrix_func(currsys, p)
        elif isinstance(trans, coord.StaticMatrixTransform):
            M = trans.matrix
        else:
            raise ValueError(
                "Transform path contains a '{0}': cannot "
                "be composed into a single transformation "
                "matrix.".format(trans.__class__.__name__)
            )

        matrices.append(M)
        currsys = p

    M = np.eye(3)
    for Mi in reversed(matrices):
        M = matrix_product(M, Mi)

    return M


# /def


##############################################################################
# END
