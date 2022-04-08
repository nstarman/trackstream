# -*- coding: utf-8 -*-

"""Processing Utilities."""


__all__ = [
    "make_timesteps",
    "make_F",
    "make_Q",
    "make_H",
    "make_R",
]


##############################################################################
# IMPORTS

# STDLIB
import warnings

# THIRD PARTY
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord, UnitSphericalRepresentation
from astropy.units import Quantity
from numpy import ndarray
from scipy.linalg import block_diag

##############################################################################
# CODE
##############################################################################


def make_timesteps(
    data: SkyCoord,
    /,
    dt0: Quantity,
    *,
    width: int = 6,
    vmin: Quantity = 0.01 * u.one,  # will error
    onsky: bool = False,
) -> ndarray:
    """Make distance arrays.

    Parameters
    ----------
    data : (N,) SkyCoord, position-only
        Must be ordered.

    N : int,  optional keyword-only
        Number of indices for convolution window. Default is 6.
    vmin : Quantity['length'] or Quantity['angle'], optional keyword-only
        Minimum distance, post-convolution. Default is 0.01

    Returns
    -------
    timesteps : (N+1,) ndarray
        Smoothed distances, starting with 0.

    Raises
    """
    # point-to-point distance
    if onsky:
        ds = data[1:].separation(data[:-1])
    elif issubclass(data.data.__class__, UnitSphericalRepresentation):
        warnings.warn("This object does not have a distance; cannot compute 3d separation.")
        ds = data[1:].separation(data[:-1])
    else:
        ds = data[1:].separation_3d(data[:-1])

    Ds = np.convolve(ds, np.ones((width,)) / width, mode="same")
    Ds[Ds < vmin] = vmin

    dts = np.insert(Ds, 0, values=dt0)
    dts = np.insert(dts, 0, values=0)

    return np.cumsum(dts.value)  # TODO! as quantity


# -------------------------------------------------------------------


def make_F(dt: float, order: int = 1, n_dims: int = 3) -> ndarray:
    """Make Transition Matrix.

    Parameters
    ----------
    dt : float
        Time step.
    order : int
        Order of transition matrix.

    Returns
    -------
    F : `~numpy.ndarray`
        Block diagonal transition matrix
    """
    # make single-component of F matrix
    if order == 1:
        f = np.array(
            [
                [1, dt],  # (position to position, position from velocity)
                [0, 1.0],  # (velocity from position, velocity to velocity)
            ],
        )
    else:
        raise NotImplementedError

    fs = [f] * n_dims  # repeat f for number of dimensions

    F: ndarray = block_diag(*fs)  # F block-diagonal array
    return F


# -------------------------------------------------------------------


# TODO check against Q_discrete_white_noise
# from filterpy.common import Q_discrete_white_noise
def make_Q(
    dt: float,
    var: float = 1.0,
    n_dims: int = 3,
    order: int = 2,
) -> ndarray:
    """Make Q Matrix.

    Parameters
    ----------
    dt : float
    var : float
    n_dims : int

    Returns
    -------
    Q : `~numpy.ndarray`

    """
    if order == 2:
        # make single-component of q matrix
        q = np.array(
            [  # single Q matrix
                [0.25 * dt ** 4, 0.5 * dt ** 3],  # 1,1 is position
                [0.5 * dt ** 3, dt ** 2],  # 2,2 is velocity
            ],
        )
    else:
        raise NotImplementedError

    qs = [q] * n_dims  # repeat q for number of dimensions

    Q: ndarray = var * block_diag(*qs)  # block diagonal stack
    return Q


def make_H(n_dims: int = 3) -> ndarray:
    """Make H Matrix.

    Parameters
    ----------
    n_dims : int, optional

    Returns
    -------
    ndarray
    """
    # component of block diagonal
    h = np.array([[0, 0], [0, 0]])
    h[0, 0] = 1

    # full matrix is for all components
    # and reduce down to `dim_z` of Kalman Filter, skipping velocity rows
    H: ndarray = block_diag(*([h] * n_dims))[::2]

    return H


def make_R(data: ndarray) -> ndarray:
    """Make R Matrix.

    Parameters
    ----------
    data : ndarray
        (len(data), dim_x)

    Returns
    -------
    R : `~numpy.ndarray`
        Diagonal array (data.shape[0], data.shape[1], data.shape[1])
        With each diagonal along axis 0 being a row from `data`.

    """
    data = np.atleast_2d(np.array(data, copy=False))
    n = data.shape[0]
    dim_x = data.shape[1]

    R = np.zeros((n, dim_x, dim_x))
    for i in range(dim_x):
        R[:, i, i] = data[:, i]

    return R


# -------------------------------------------------------------------


# def p2p_distance(X: Sequence, axis: int = 0) -> Sequence:
#     """Point-to-Point distance in Cartesian coordinates.
#
#     Parameters
#     ----------
#     X : array
#         shape N data rows, by M coordinate columns
#         The array must be correctly sorted.
#     axis : int
#         The axis on which the distance is computed:
#
#         - 0 is comparing rows (so each row is the full set of coordinates)
#         - 1 is comparing columns (so each row is one coordinate type)
#
#     Returns
#     -------
#     full_arc : Sequence
#         Arc length from 1st point (inclusive) to last point in `X`
#         length `X` along `axis`
#
#     """
#     ds = np.linalg.norm(np.diff(X, axis=axis), axis=axis - 1)
#     arc = np.cumsum(ds)
#     full_arc = np.insert(arc, 0, 0)
#
#     return full_arc
