# -*- coding: utf-8 -*-

"""Get tracks."""

__author__ = "Nathaniel Starkman"
__credit__ = ["Jo Bovy", "Jeremy Webb"]


__all__ = ["p2p_distance_cartesian", "p2p_distance_spherical"]


##############################################################################
# IMPORTS

# BUILT-IN

import typing as T


# THIRD PARTY

import numpy as np


# PROJECT-SPECIFIC


##############################################################################
# PARAMETERS


##############################################################################
# CODE
##############################################################################


def p2p_distance_cartesian(X: T.Sequence, axis: int = 0) -> T.Sequence:
    """Point-to-Point distance in Cartesian coordinates.

    Parameters
    ----------
    X : array
        shape N data rows, by M coordinate columns
        The array must be correctly sorted.
    axis : int
        The axis on which the distance is computed:

        - 0 is comparing rows (so each row is the full set of coordinates)
        - 1 is comparing columns (so each row is one coordinate type)

    Returns
    -------
    full_arc : Sequence
        Arc length from 1st point (inclusive) to last point in `X`
        length `X` along `axis`

    """
    ds = np.linalg.norm(np.diff(X, axis=axis), axis=axis - 1)
    arc = np.cumsum(ds)
    full_arc = np.insert(arc, 0, 0)

    return full_arc


# /def


# -------------------------------------------------------------------


def p2p_distance_spherical(X):
    """Angular separation between two points on a sphere.

    .. todo::

        support axis argument so can have shape (2, N)

    Parameters
    ----------
    X : :class:`~numpy.ndarray` or `~astropy.units.Quantity`
        Longitude and latitude of the two points.
        Must be in radians.
        Shape (N, 2)

    Returns
    -------
    angular_separation : :class:`~numpy.ndarray` of float
        "units" depends on input `rad`

    Notes
    -----
    The angular separation is calculated using the Vincenty formula [1]_,
    which is slightly more complex and computationally expensive than
    some alternatives, but is stable at at all distances, including the
    poles and antipodes.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Great-circle_distance

    See Also
    --------
    :func:`~astropy.coordinates.angle_utilities.angular_separation`

    """
    lon1 = X[:-1, 0]
    lon2 = X[1:, 0]

    lat1 = X[:-1, 1]
    lat2 = X[1:, 1]

    sdlon = np.sin(lon2 - lon1)
    cdlon = np.cos(lon2 - lon1)
    slat1 = np.sin(lat1)
    slat2 = np.sin(lat2)
    clat1 = np.cos(lat1)
    clat2 = np.cos(lat2)

    num1 = clat2 * sdlon
    num2 = clat1 * slat2 - slat1 * clat2 * cdlon
    denominator = slat1 * slat2 + clat1 * clat2 * cdlon

    return np.arctan2(np.hypot(num1, num2), denominator)


# /def


##############################################################################
# END
