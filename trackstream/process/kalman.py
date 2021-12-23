# -*- coding: utf-8 -*-

"""Kalman Filter code."""

__all__ = ["KalmanFilter"]


##############################################################################
# IMPORTS

# STDLIB
import typing as T
import warnings
from collections import namedtuple

# THIRD PARTY
import numpy as np

# LOCAL
from . import utils
from .core import predict as custom_predict
from .core import rts_smoother as custom_rts_smoother
from .core import update as custom_update
from trackstream.config import conf
from trackstream.setup_package import HAS_FILTERPY

if HAS_FILTERPY:
    # THIRD PARTY
    from filterpy.kalman import predict as filterpy_predict
    from filterpy.kalman import rts_smoother as filterpy_rts_smoother
    from filterpy.kalman import update as filterpy_update

##############################################################################
# PARAMETERS

kalman_output = namedtuple(
    "kalman_output",
    field_names=["Xs", "Ps", "Fs", "Qs"],
)

##############################################################################
# CODE
##############################################################################


class KalmanFilter:
    """KalmanFilter class.

    .. todo::

        allow these to be callable

    Parameters
    ----------
    x0 : np.ndarray
        Initial position
    F0 : :class:`numpy.ndarray` or None (optional)
        Transition Matrix.
    Q : :class:`numpy.ndarray` or None (optional)
    H : :class:`numpy.ndarray` or None (optional)
    R : :class:`numpy.ndarray` or None (optional)
    P : :class:`numpy.ndarray` or None (optional)

    """

    def __init__(
        self,
        x0: np.ndarray,
        P0: np.ndarray,
        F0: T.Union[np.ndarray, T.Callable[[float], np.ndarray]] = None,
        Q0: T.Optional[np.ndarray] = None,
        H0: T.Optional[np.ndarray] = None,
        R0: T.Optional[np.ndarray] = None,
        **kwargs
    ):
        super().__init__()

        self.x0 = x0
        self.P0 = P0

        self.F0 = F0
        self.Q0 = Q0
        self.H0 = H0
        self.R0 = R0

        self.options = kwargs

    # /def

    #######################################################
    # Run

    def fit(
        self,
        data: T.Union[np.ndarray, T.Sequence],
        dts: np.ndarray,
        *,
        method: T.Literal["stepupdate"],
        use_filterpy: T.Optional[bool] = None,
        **kwargs
    ) -> T.Tuple[kalman_output, kalman_output]:
        """Run Kalman Filter with updates on each step.



        Parameters
        ----------
        data : |ndarray|
        dts: |ndarray|
        method : {"stepupdate"}
            Which method to use.

            - "stepupdate" : ``fit_with_stepupdate``


        use_filterpy : bool or None, (optional, keyword-only)
            If none, uses configuration.

        **kwargs
            Passed to fit method.

        Returns
        -------
        `~kalman_output`
            "Xs", "Ps", "Fs", "Qs"
        """
        kw = dict(self.options.items())  # copy
        kw.update(kwargs)

        if method == "stepupdate":
            result = self.fit_with_stepupdate(
                data, dts, use_filterpy=use_filterpy, full_output=False, **kw
            )

        else:
            raise ValueError()

        return result

    # /def

    def fit_with_stepupdate(
        self,
        data: T.Union[np.ndarray, T.Sequence],
        dts: np.ndarray,
        u: T.Union[np.ndarray, float] = 0.0,
        B: T.Union[np.ndarray, float] = 1.0,
        alpha: float = 1.0,
        *,
        full_output: bool = False,
        use_filterpy: T.Optional[bool] = None,
        q_kw: T.Optional[T.Dict] = None
    ) -> T.Tuple[kalman_output, kalman_output]:
        """Run Kalman Filter with updates on each step.

        Parameters
        ----------
        data : |ndarray|
        dts: |ndarray|
        u : |ndarray| or float (optional)
            Default is 0.0
        B : |ndarray| or float (optional)
            Default is 0
        alpha : float (optional)
            Default is 1.0
        use_filterpy : bool or None, (optional, keyword-only)
            If none, uses configuration.

        Returns
        -------
        output : `~kalman_output`
            "Xs", "Ps", "Fs", "Qs"
        smooth : `~kalman_output`
            "Xs", "Ps", "Fs", "Qs"

        """
        if use_filterpy is None:
            use_filterpy = conf.use_filterpy

        if use_filterpy and HAS_FILTERPY:
            updater = filterpy_update
            predicter = filterpy_predict
            smoother = filterpy_rts_smoother
        else:
            updater = custom_update
            predicter = custom_predict
            smoother = custom_rts_smoother

            if use_filterpy:
                warnings.warn("can't use filterpy.")

        # /if

        q_kw = q_kw or {}  # None -> dict

        # starting points
        x = self.x0
        P = self.P0

        R = self.R0  # TODO! make function of data point!
        H = self.H0

        if callable(self.F0):
            F = self.F0(dts[0])
            make_F = self.F0
        else:
            F = self.F0
            make_F = utils.make_F

        if callable(self.Q0):
            Q = self.Q0(dts[0])
            make_Q = self.Q0
        else:
            Q = self.Q0
            make_Q = utils.make_Q

        n = len(data)
        # initialize arrays
        Xs = np.empty((n, *np.shape(x)))
        Ps = np.empty((n, *np.shape(P)))
        Fs = np.empty((n, *np.shape(F)))
        Qs = np.empty((n, *np.shape(Q)))
        # iterate predict & update steps
        for i, z in enumerate(data):
            # F, Q
            F = make_F(dts[i])
            Q = make_Q(dts[i], **q_kw)

            # predict & update
            x, P = predicter(x, P=P, F=F, Q=Q, u=u, B=B, alpha=alpha)
            x, P, *_ = updater(x, P=P, z=z, R=R, H=H)

            # append results
            Xs[i], Ps[i] = x, P
            Fs[i], Qs[i] = F, Q

        # /for

        # recast as arrays
        Xs, Ps = np.array(Xs), np.array(Ps)
        Fs, Qs = np.array(Fs), np.array(Qs)

        # smoothed
        sXs, sPs, sFs, sQs = smoother(Xs, Ps, Fs, Qs)
        smooth = kalman_output(sXs, sPs, sFs, sQs)

        if full_output:
            output = kalman_output(Xs, Ps, Fs, Qs)
            return smooth, output

        return smooth

    # =================
    # Misc

    @staticmethod
    def make_simple_dts(
        ordered_data: np.ndarray, dt0: float, *, N: int = 6, vmin: float = 0.01, axis: int = 1
    ):
        dts = utils.make_dts(
            ordered_data=ordered_data,
            dt0=dt0,
            N=N,
            vmin=vmin,
            axis=axis,
        )

        return dts


##############################################################################
# END

if conf.use_filterpy and not HAS_FILTERPY:
    warnings.warn("filterpy not installed. Will use built-in.")
