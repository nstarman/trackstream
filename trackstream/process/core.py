# -*- coding: utf-8 -*-

"""Processing by Kalman filter."""

__all__ = [
    # functions
    "predict",
    "update",
    "rts_smoother",
]


##############################################################################
# IMPORTS

# STDLIB
import typing as T

# THIRD PARTY
import numpy as np
from numpy import dot, linalg
from scipy.stats import multivariate_normal

##############################################################################
# CODE
##############################################################################


def predict(
    x: np.ndarray,
    P: np.ndarray,
    F: np.ndarray,
    Q: np.ndarray,
    u: T.Union[np.ndarray, float] = 0.0,
    B: T.Union[np.ndarray, float] = 1.0,
    alpha: float = 1.0,
) -> T.Tuple[np.ndarray, np.ndarray]:
    """Predict prior using Kalman filter transition functions.

    Parameters
    ----------
    x : |ndarray|
        State.
    P : |ndarray|
        Covariance matrix for state.
    F : |ndarray|
        Transition matrix.
    Q : |ndarray|, optional
        Process noise matrix. Gets added to covariance matrix.
    u : |ndarray|, optional
        Control vector.
    B : |ndarray|, optional
        Control transition matrix.
    alpha : float, optional
        Fading memory parameter. Default (1.0) is a Kalman filter.

    Returns
    -------
    x : |ndarray|
        Prior state vector. The 'predicted' state.
    P : |ndarray|
        Prior covariance matrix. The 'predicted' covariance.

    """
    # predict position
    x = dot(F, x) + dot(B, u)
    # & covariance matrix
    P = (alpha ** 2) * dot(F, dot(P, F.T)) + Q

    return x, P


# /def


# -------------------------------------------------------------------


def update(
    x: np.ndarray, P: np.ndarray, z: np.ndarray, R: np.ndarray, H: np.ndarray, **kw
) -> T.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,]:
    """Add a new measurement (z) to the Kalman filter.

    Implemented to be compatible with `filterpy`.

    .. |ndarray| replace:: `~numpy.ndarray`

    Parameters
    ----------
    x : |ndarray|
        State estimate vector. Dimensions (x, 1).
    P : |ndarray|
        Covariance matrix. Dimensions (x, x)
    z : |ndarray|
        Measurement. Dimensions (z, 1)
    R : |ndarray|
        Measurement noise matrix. Dimensions (z, z)
    H : |ndarray|
        Measurement function.  Dimensions (x, x)

    kw : Any
        Absorbs extra arguments for ``filterpy`` compatibility.

    Returns
    -------
    x : |ndarray|
        Posterior state estimate.
    P : |ndarray|
        Posterior covariance matrix.
    resid : |ndarray|
        Residual between measurement and prediction
    K : |ndarray|
        Kalman gain
    S : |ndarray|
        System uncertainty in measurement space.
    log_lik : |ndarray|
        Log-likelihood. A multivariate normal.

    """
    S = dot(dot(H, P), H.T) + R  # system uncertainty in measurement space
    K = dot(dot(P, H.T), linalg.inv(S))  # Kalman gain

    predict = dot(H, x)  # prediction in measurement space
    resid = z - predict  # measurement and prediction residual

    x = x + dot(K, resid)  # predict new x using Kalman gain

    KH = dot(K, H)
    ImKH = np.eye(KH.shape[0]) - KH
    # stable representation from Filterpy
    # P = (1 - KH)P(1 - KH)' + KRK'
    P = dot(dot(ImKH, P), ImKH.T) + dot(dot(K, R), K.T)

    # log-likelihood
    log_lik = multivariate_normal.logpdf(
        z.flatten(),
        mean=dot(H, x).flatten(),
        cov=S,
        allow_singular=True,
    )

    return x, P, resid, K, S, log_lik


# /def


# -------------------------------------------------------------------
# Run Rauch-Tung-Striebel Filter


def rts_smoother(
    Xs: np.ndarray,
    Ps: np.ndarray,
    Fs: T.Union[T.Sequence[np.ndarray], np.ndarray],
    Qs: T.Union[T.Sequence[np.ndarray], np.ndarray],
) -> T.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run Rauch-Tung-Striebel Kalman smoother on Kalman filter series.

    Implemented to be compatible with `filterpy`.

    .. |ndarray| replace:: `~numpy.ndarray`

    Parameters
    ----------
    Xs : |ndarray|
       array of the means (state variable x) of the output of a Kalman
       filter.
    Ps : |ndarray|
        array of the covariances of the output of a kalman filter.
    Fs : Sequence of |ndarray|
        State transition matrix of the Kalman filter at each time step.
    Qs : Sequence of |ndarray|
        Process noise of the Kalman filter at each time step.

    Returns
    -------
    x : |ndarray|
       Smoothed state.
    P : |ndarray|
       Smoothed state covariances.
    K : |ndarray|
        Smoother gain along x.
    Ppred : |ndarray|
       predicted state covariances

    """
    # copy
    x = Xs.copy()
    P = Ps.copy()
    Ppred = Ps.copy()

    # initialize parameters
    n, dim_x = Xs.shape
    K = np.zeros((n, dim_x, dim_x))

    # iterate through, running Kalman system again.
    for i in reversed(range(n - 1)):  # [n-2, ..., 0]
        # prediction
        Ppred[i] = dot(dot(Fs[i], P[i]), Fs[i].T) + Qs[i]
        # Kalman gain
        K[i] = dot(dot(P[i], Fs[i].T), linalg.inv(Ppred[i]))

        # update position and covariance
        xprior = dot(Fs[i], x[i])
        x[i] = x[i] + dot(K[i], x[i + 1] - xprior)
        P[i] = P[i] + dot(dot(K[i], P[i + 1] - Ppred[i]), K[i].T)

    # /for

    return x, P, K, Ppred


# /def


##############################################################################
# END
