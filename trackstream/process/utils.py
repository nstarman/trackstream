# -*- coding: utf-8 -*-

"""Processing Utilities."""


__all__ = [
    "make_dts",
    "make_F",
    "make_Q",
    "make_H",
    "make_R",
    "p2p_distance",
]


##############################################################################
# IMPORTS

# STDLIB
import typing as T

# THIRD PARTY
import numpy as np
from scipy.linalg import block_diag

# LOCAL
from .plot import plot_dts

##############################################################################
# CODE
##############################################################################


def make_dts(
    ordered_data: np.ndarray,
    dt0: float,
    *,
    N: int = 6,
    vmin: float = 0.01,
    axis: int = 1,
    plot: bool = False
) -> np.ndarray:
    """Make distance arrays.

    Parameters
    ----------
    ordered_data : array-like
    dt0 : float
        Initial time-step

    N : int
        Number of indices for convolution window.
    vmin : float
        Minimum distance, post-convolution.
    axis : int
        axis in `ordered_data` over which to find point-to-point
        distance.
    plot : bool
        Whether to plot.

    Returns
    -------
    dts : ndarray
        Smoothed distance.
    """
    ds = np.linalg.norm(
        ordered_data[1:] - ordered_data[:-1],
        axis=axis,
    )  # point-to-point distance

    Ds = np.convolve(ds, np.ones((N,)) / N, mode="same")
    Ds[Ds < vmin] = vmin

    dts = np.insert(Ds, 0, values=dt0)

    if plot:
        plot_dts(ds, Ds)

    # /if

    return dts


# -------------------------------------------------------------------


def make_F(dt: float, order: int = 1, n_dims=3) -> np.ndarray:
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

    F = block_diag(*fs)  # F block-diagonal array
    return F


# -------------------------------------------------------------------


# TODO check against Q_discrete_white_noise
# from filterpy.common import Q_discrete_white_noise
def make_Q(
    dt: float,
    var: float = 1.0,
    n_dims: int = 3,
    order: int = 2,
) -> np.ndarray:
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
        NotImplementedError

    qs = [q] * n_dims  # repeat q for number of dimensions

    Q = var * block_diag(*qs)  # block diagonal stack
    return Q


# -------------------------------------------------------------------


def make_H(position_only=True, n_dims: int = 3, order: int = 1) -> np.ndarray:
    """Make H Matrix.

    Parameters
    ----------
    position_only
    n_dims
    order

    Returns
    -------
    H : `~numpy.ndarray`

    """
    if order == 1:
        h = np.array([[0, 0], [0, 0]])
    else:
        raise NotImplementedError

    if position_only:
        h[0, 0] = 1

        # full matrix is for all components
        # and reduce down to `dim_z` of Kalman Filter, skipping velocity rows
        H = block_diag(*([h] * n_dims))[:: (order + 1)]

    else:
        raise NotImplementedError

    return H


# -------------------------------------------------------------------


def make_R(data) -> np.ndarray:
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


def p2p_distance(X: T.Sequence, axis: int = 0) -> T.Sequence:
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


##############################################################################
# END
