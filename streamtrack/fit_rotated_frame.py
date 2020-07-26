# -*- coding: utf-8 -*-


"""Fit a Rotated ICRS reference frame."""


__all__ = [
    # functions
    "cartesian_model",
    # other
    "residual",
]


##############################################################################
# IMPORTS

# BUILT-IN

# THIRD PARTY

import numpy as np

from utilipy.data_utils.fitting import scipy_residual_to_lmfit


# PROJECT-SPECIFIC

from .utils import reference_to_skyoffset_matrix, cartesian_to_spherical


##############################################################################
# PARAMETERS


##############################################################################
# CODE
##############################################################################


def cartesian_model(data, *, lon, lat, rotation, deg=True):
    """Model from Cartesian Coordinates.

    .. |AngleType| replace:: :class:`~astropy.coordinates.Angle`
    .. |Quantity| replace:: :class:`~astropy.units.Quantity`

    Parameters
    ----------
    x, y, z : array_like
    lon, lat : float or |AngleType| or |Quantity| instance
        The longitude and latitude origin for the reference frame.
        If float, assumed degrees.
    rotation : float or |AngleType| or |Quantity| instance
        The final rotation of the frame about the ``origin``. The sign of
        the rotation is the left-hand rule.  That is, an object at a
        particular position angle in the un-rotated system will be sent to
        the positive latitude (z) direction in the final frame.
        If float, assumed degrees.

    Returns
    -------
    r, lat, lon : array_like
        Same shape as `x`, `y`, `z`.

    Other Parameters
    ----------------
    deg : bool
        whether to return `lat` and `lon` as degrees
        (default True) or radians.

    """
    rot_matrix = reference_to_skyoffset_matrix(lon, lat, rotation)
    rot_xyz = np.dot(rot_matrix, data.xyz.value).reshape(-1, len(data))

    r, lat, lon = cartesian_to_spherical(*rot_xyz, deg=deg)

    return r, lon, lat


# /def


# -------------------------------------------------------------------


@scipy_residual_to_lmfit.decorator(param_order=["rotation", "lon", "lat"])
def residual(variables, data, scalar=False):
    r"""How close rotated dec (phi2) is to flat.

    Parameters
    ----------
    variables : Sequence[float]
        (rotation, lon, lat)

        - rotation angle : float
            The final rotation of the frame about the ``origin``. The sign of
            the rotation is the left-hand rule.  That is, an object at a
            particular position angle in the un-rotated system will be sent to
            the positive latitude (z) direction in the final frame.
            In degrees.
        - lon, lat : float
            In degrees. If ICRS equivalent to ra & dec.
    data : Sequence
        The data in cartesian xyz form.
        eg. ``ICRS.cartesian.xyz.value``

    Returns
    -------
    res : float or Sequence
        :math:`\rm{lat} - 0`

        If `scalar` is True, then sum array_like to return float.

    Other Parameters
    ----------------
    scalar : bool
        Whether to sum `res` into a float.
        Note that if `res` is also a float, it is unaffected.

    """
    rotation = variables[0]
    lon = variables[1]
    lat = variables[2]

    r, lon, lat = cartesian_model(
        data, lon=lon, lat=lat, rotation=rotation, deg=True
    )

    res = np.abs(lat - 0.0)  # phi2 - 0

    if scalar:
        return np.sum(res)
    return res


# /def


# -------------------------------------------------------------------


##############################################################################
# END
