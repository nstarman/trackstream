# -*- coding: utf-8 -*-

"""Preprocessing.

TODO find slowest lines
https://marcobonzanini.com/2015/01/05/my-python-code-is-slow-tips-for-profiling/

"""


__all__ = ["batch_predict_with_stepupdate"]


##############################################################################
# IMPORTS

# BUILT-IN

# BUILT-IN
import typing as T
import warnings
from collections import namedtuple

# THIRD PARTY
import numpy as np

# try to import optional dependency `filterpy`
try:
    # THIRD PARTY
    from filterpy.kalman import predict as filterpy_predict
    from filterpy.kalman import rts_smoother as filterpy_rts_smoother
    from filterpy.kalman import update as filterpy_update
except ImportError:
    _HAS_FILTERPY = False
else:
    _HAS_FILTERPY = True


# PROJECT-SPECIFIC

from ..conf import conf
from .kalman import predict, rts_smoother, update
from .utils import make_F, make_Q

if conf.use_filterpy and not _HAS_FILTERPY:
    warnings.warn("filterpy not installed. Will use built-in.")


##############################################################################
# PARAMETERS

run_output = namedtuple("run_output", field_names=["Xs", "Ps", "Fs", "Qs"])


##############################################################################
# CODE
##############################################################################


def batch_predict_with_stepupdate(
    data: T.Union[np.ndarray, T.Sequence],
    dts: np.ndarray,
    x0: np.ndarray,
    P: np.ndarray,
    R: np.ndarray,
    H: np.ndarray,
    u: T.Union[np.ndarray, float] = 0.0,
    B: T.Union[np.ndarray, float] = 1.0,
    alpha: float = 1.0,
    *,
    use_filterpy: T.Optional[bool] = None,
    qkw: T.Any = dict()
):
    """Run Kalman Filter with updates on each step.

    .. |ndarray| replace:: `~numpy.ndarray`

    Parameters
    ----------
    data : |ndarray|
    dts: |ndarray|
    x0: |ndarray|
    P: |ndarray|
    R: |ndarray|
    H : |ndarray|
    u : |ndarray| or float, optional
        Default is 0.0
    B : |ndarray| or float, optional
        Default is 0
    alpha : float, optional
        Default is 1.0
    use_filterpy : bool or None, optional, kwarg only
        If none, uses configuration.

    Returns
    -------
    output : `~run_output`
        "Xs", "Ps", "Fs", "Qs"
    smooth : `~run_output`
        "Xs", "Ps", "Fs", "Qs"

    """
    if use_filterpy is None:
        use_filterpy = conf.use_filterpy

    if use_filterpy and _HAS_FILTERPY:
        updater = filterpy_update
        predicter = filterpy_predict
        smootherRTS = filterpy_rts_smoother
    else:
        updater = update
        predicter = predict
        smootherRTS = rts_smoother

    # /if

    x = x0  # starting point
    # n = len(data)
    # initialize arrays
    Xs, Ps, Fs, Qs = [], [], [], []
    # iterate predict & update steps
    for i, z in enumerate(data):
        # F, Q
        F = make_F(dts[i])
        Q = make_Q(dts[i], **qkw)

        # predict & update
        x, P = predicter(x, P=P, F=F, Q=Q, u=u, B=B, alpha=alpha)
        x, P, *_ = updater(x, P=P, z=z, R=R, H=H)

        # append results
        Xs.append(x)
        Ps.append(P)
        Fs.append(F)
        Qs.append(Q)

    # /for

    Xs, Ps = np.array(Xs), np.array(Ps)
    Fs, Qs = np.array(Fs), np.array(Qs)

    # smooth
    sXs, sPs, sFs, sQs = smootherRTS(Xs, Ps, Fs, Qs)

    # make namedtuples
    output = run_output(Xs, Ps, Fs, Qs)
    smooth = run_output(sXs, sPs, sFs, sQs)

    return output, smooth


# /def


# -------------------------------------------------------------------


##############################################################################
# END
