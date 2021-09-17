# -*- coding: utf-8 -*-
# see LICENSE.rst

"""Examples Coordinate Utilities."""


__all__ = [
    "get_transform_matrix",
]


##############################################################################
# IMPORTS

# STDLIB
import typing as T

# THIRD PARTY
import astropy.coordinates as coord
import numpy as np
from astropy.coordinates.matrix_utilities import matrix_product

# LOCAL
from trackstream._type_hints import CoordinateType

##############################################################################
# PARAMETERS

##############################################################################
# CODE
##############################################################################


def get_transform_matrix(
    from_frame: CoordinateType,
    to_frame: CoordinateType,
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
        from_frame_cls,
        to_frame_cls,
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
                "matrix.".format(trans.__class__.__name__),
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
