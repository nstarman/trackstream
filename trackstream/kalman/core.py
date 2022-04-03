# -*- coding: utf-8 -*-

"""Kalman Filter code."""

__all__ = ["FirstOrderNewtonianKalmanFilter"]


##############################################################################
# IMPORTS

# STDLIB
import warnings
from types import FunctionType
from typing import Any, Callable, Dict, Literal, NamedTuple, Optional, Sequence, Tuple, Union

# THIRD PARTY
import astropy.units as u
import numpy as np
from astropy.coordinates import BaseCoordinateFrame, BaseRepresentation, SkyCoord
from astropy.units import Quantity
from numpy import array, dot, linalg, ndarray
from scipy.stats import multivariate_normal

# LOCAL
from . import helper
from trackstream.base import CommonBase
from trackstream.config import conf
from trackstream.setup_package import HAS_FILTERPY
from trackstream.utils.misc import intermix_arrays
from trackstream.utils.path import Path, path_moments

if HAS_FILTERPY:
    # THIRD PARTY
    from filterpy.kalman import predict as filterpy_predict
    from filterpy.kalman import rts_smoother as filterpy_rts_smoother
    from filterpy.kalman import update as filterpy_update

##############################################################################
# PARAMETERS


class kalman_output(NamedTuple):
    timesteps: ndarray
    Xs: ndarray
    Ps: ndarray
    Qs: ndarray


##############################################################################
# CODE
##############################################################################


class FirstOrderNewtonianKalmanFilter(CommonBase):
    """First-order Newtonian Kalman filter class.

    Parameters
    ----------
    x0 : ndarray
        Initial position.
    P0 : ndarray
        Initial covariance.

    frame : BaseCoordinateFrame, keyword-only
        The frame of the data, without
    kfv0 : ndarray
        Initial filter 'velocity'.

    F0 : ndarray callable or None, optional keyword-only
        State transition matrix.
    Q0 : ndarray callable or None, optional keyword-only
    R0 : ndarray callable or None, optional keyword-only
    """

    def __init__(
        self,
        x0: ndarray,
        P0: ndarray,
        *,
        frame: BaseCoordinateFrame,
        representation_type: Optional[BaseRepresentation] = None,
        kfv0: Optional[ndarray] = None,
        F0: Union[None, ndarray, Callable[[float], ndarray]] = None,
        Q0: Optional[ndarray] = None,
        R0: ndarray,
        **kwargs: Any,
    ) -> None:
        super().__init__(frame=frame, representation_type=representation_type)
        self._options = kwargs

        ndims = len(x0)
        if ndims < 2 or ndims > 6:
            raise ValueError

        # TODO! x0 as a Representation (or higher) object
        _kfv0: ndarray = array([0] * ndims) if kfv0 is None else kfv0

        # Initial state
        self._x0 = intermix_arrays(x0, _kfv0)
        self._P0 = P0  # TODO! make optional
        self._H = helper.make_H(n_dims=ndims)
        self._F0 = F0 if not callable(F0) else None
        self._Q0 = Q0 if not callable(Q0) else None
        self._R0 = R0  # TODO! make function

        # functions
        self._state_transition_model = F0 if callable(F0) else helper.make_F
        self._process_noise_model = Q0 if callable(Q0) else helper.make_Q

        # Running
        self.__result: Optional[kalman_output] = None
        self.__smooth_result: Optional[kalman_output] = None

    # ---------------------------------------------------------------

    @property
    def x0(self) -> ndarray:
        """Initial state."""
        return self._x0

    @property
    def P0(self) -> ndarray:
        """Initial state covariance matrix."""
        return self._P0

    @property
    def F0(self) -> Optional[ndarray]:
        """Initial state-transition model."""
        return self._F0

    @property
    def Q0(self) -> Optional[ndarray]:
        return self._Q0

    @property
    def H(self) -> ndarray:
        return self._H

    @property
    def R0(self) -> ndarray:
        return self._R0

    @property
    def options(self) -> Dict[str, Any]:
        return self._options

    @property
    def state_transition_model(self) -> Any:  # TODO!
        return self._state_transition_model

    @property
    def process_noise_model(self) -> Callable:
        return self._process_noise_model

    # ---------------------------------------------------------------
    # Requires the Kalman filter to be run
    # TODO better error type than ValueError

    @property
    def _result(self) -> kalman_output:
        if self.__result is None:
            raise ValueError(f"need to run {self.__class__.__qualname__}.fit()")
        return self.__result

    @property
    def _smooth_result(self) -> kalman_output:
        if self.__smooth_result is None:
            raise ValueError(f"need to run {self.__class__.__qualname__}.fit()")
        return self.__smooth_result

    #######################################################
    # Math (2 phase + smoothing)

    def _get_math_methods(
        self, use_filterpy: Optional[bool]
    ) -> Tuple[Callable, Callable, Callable]:
        """Get the math methods."""
        if use_filterpy is None:
            use_filterpy = conf.use_filterpy

        if use_filterpy and HAS_FILTERPY:
            predicter = filterpy_predict
            updater = filterpy_update
            smoother = filterpy_rts_smoother
        else:
            predicter = self._math_predict
            updater = self._math_update
            smoother = self._math_smoother

            if use_filterpy:
                warnings.warn("can't use filterpy.")

        return predicter, updater, smoother

    def _math_predict(
        self, x: ndarray, P: ndarray, F: ndarray, Q: ndarray, **kw: Any
    ) -> Tuple[ndarray, ndarray]:
        """Predict prior using Kalman filter transition functions.

        Parameters
        ----------
        x : ndarray
            State.
        P : ndarray
            Covariance matrix for state.
        F : ndarray
            Transition matrix.
        Q : ndarray, optional
            Process noise matrix. Gets added to covariance matrix.

        Returns
        -------
        x : ndarray
            Prior state vector. The 'predicted' state.
        P : ndarray
            Prior covariance matrix. The 'predicted' covariance.
        """
        # predict position (not including control matrix)
        x = F @ x
        # & covariance matrix
        P = F @ P @ F.T + Q

        return x, P

    def _math_update(
        self, x: ndarray, P: ndarray, z: ndarray, R: ndarray, H: ndarray, **kw: Any
    ) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
        """Add a new measurement (z) to the Kalman filter.

        Implemented to be compatible with `filterpy`.

        Parameters
        ----------
        x : ndarray
            State estimate vector. Dimensions (x, 1).
        P : ndarray
            Covariance matrix. Dimensions (x, x)
        z : ndarray
            Measurement. Dimensions (z, 1)
        R : ndarray
            Measurement noise matrix. Dimensions (z, z)
        H : ndarray
            Measurement function.  Dimensions (x, x)

        **kw : Any
            Absorbs extra arguments for ``filterpy`` compatibility.

        Returns
        -------
        x : ndarray
            Posterior state estimate.
        P : ndarray
            Posterior covariance matrix.
        resid : ndarray
            Residual between measurement and prediction
        K : ndarray
            Kalman gain
        S : ndarray
            System uncertainty in measurement space.
        log_lik : ndarray
            Log-likelihood. A multivariate normal.
        """
        S = H @ P @ H.T + R  # system uncertainty in measurement space
        K = (P @ H.T) @ linalg.inv(S)  # Kalman gain

        predict = H @ x  # prediction in measurement space
        resid = z - predict  # measurement and prediction residual

        x = x + K @ resid  # predict new x using Kalman gain

        KH = K @ H
        ImKH = np.eye(KH.shape[0]) - KH
        # stable representation from Filterpy
        # P = (1 - KH)P(1 - KH)' + KRK'
        P = ImKH @ P @ ImKH.T + K @ R @ K.T

        # log-likelihood
        log_lik = multivariate_normal.logpdf(
            z.flatten(),
            mean=dot(H, x).flatten(),
            cov=S,
            allow_singular=True,
        )

        return x, P, resid, K, S, log_lik

    def _math_smoother(
        self,
        Xs: ndarray,
        Ps: ndarray,
        Fs: Union[Sequence[ndarray], ndarray],
        Qs: Union[Sequence[ndarray], ndarray],
    ) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        """Run Rauch-Tung-Striebel Kalman smoother on Kalman filter series.

        Implemented to be compatible with `filterpy`.

        Parameters
        ----------
        Xs : ndarray
           array of the means (state variable x) of the output of a Kalman
           filter.
        Ps : ndarray
            array of the covariances of the output of a kalman filter.
        Fs : Sequence of ndarray
            State transition matrix of the Kalman filter at each time step.
        Qs : Sequence of ndarray
            Process noise of the Kalman filter at each time step.

        Returns
        -------
        x : ndarray
           Smoothed state.
        P : ndarray
           Smoothed state covariances.
        K : ndarray
            Smoother gain along x.
        Ppred : ndarray
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
            Ppred[i] = Fs[i] @ P[i] @ Fs[i].T + Qs[i]
            # Kalman gain
            K[i] = P[i] @ Fs[i].T @ linalg.inv(Ppred[i])

            # update position and covariance
            xprior = Fs[i] @ x[i]
            x[i] = x[i] + K[i] @ (x[i + 1] - xprior)
            P[i] = P[i] + K[i] @ (P[i + 1] - Ppred[i]) @ K[i].T

        return x, P, K, Ppred

    #######################################################
    # Fit

    def fit(
        self,
        data: SkyCoord,
        /,
        timesteps: ndarray,
        *,
        use_filterpy: Optional[bool] = None,
        **kwargs: Any,
    ) -> Tuple[kalman_output, Path]:
        """Run Kalman Filter with updates on each step.

        Parameters
        ----------
        data : (N,D) ndarray, position-only
        timesteps: (N+1,) ndarray
            Must be start and end-point inclusive.

        use_filterpy : bool or None, optional keyword-only
            If none, uses configuration.
        **kwargs
            Passed to fit method.

        Returns
        -------
        smooth : `~kalman_output`
            "timesteps", "Xs", "Ps", "Qs"
        """
        kw = {**self.options, **kwargs}  # copy

        dts = np.diff(timesteps)
        if len(dts) != len(data):
            raise ValueError(f"len(timesteps)={len(timesteps)} is not {len(data)+1}")

        # Kalman filter math methods
        predicter, updater, smoother = self._get_math_methods(use_filterpy)

        # ------ setup ------
        self._original_data = data  # save the data used to fit
        self._timesteps = timesteps  # save the timesteps
        # TODO! check timesteps start with 0

        # starting points
        x, P = self.x0, self.P0
        R = self.R0  # TODO! make function of data point!

        F = self.state_transition_model(dts[0]) if self.F0 is None else self.F0
        Q = self.process_noise_model(dts[0]) if self.Q0 is None else self.Q0

        n = len(data)
        # initialize arrays
        Xs = np.empty((n, *np.shape(x)))
        Ps = np.empty((n, *np.shape(P)))
        Fs = np.empty((n, *np.shape(F)))
        Qs = np.empty((n, *np.shape(Q)))

        # iterate predict & update steps
        for i, (z, dt) in enumerate(zip(data, dts)):
            # F, Q
            F = self.state_transition_model(dt)
            Q = self.process_noise_model(dt, **kwargs)

            # predict & update
            x, P = predicter(x, P=P, F=F, Q=Q)
            x, P, *_ = updater(x, P=P, z=z, R=R, H=self._H)

            # append results
            Xs[i], Ps[i] = x, P
            Qs[i] = Q

        # save
        self.__result = kalman_output(timesteps, Xs, Ps, Qs)

        # smoothed
        try:
            xs, Ps, _, Qs = smoother(Xs, Ps, Fs, Qs)

        except Exception:
            print("FIXME!")
            smooth = self._result
        else:
            smooth = kalman_output(timesteps, xs, Ps, Qs)
            self.__smooth_result = smooth

        # TODO! make sure get the frame and units right
        r = self.representation_type(smooth.Xs[:, ::2].T, unit=u.kpc)
        c = self.frame.realize_frame(r)  # (not interpolated)

        # covariance matrix. select only the phase-space positions
        # everything is Gaussian so there are no off-diagonal elements,
        # so the 1-sigma error is quite easy.
        cov = smooth.Ps[:, ::2, ::2]  # TODO! use H
        var = np.diagonal(cov, axis1=1, axis2=2)
        sigma = np.sqrt(np.sum(np.square(var), axis=-1)) * u.kpc

        # TODO! is this the Affine wanted?
        sp2p = c[:-1].separation(c[1:])  # point-2-point sep
        affine = np.concatenate(([min(1e-10 * sp2p.unit, 1e-10 * sp2p[0])], sp2p.cumsum()))

        self._path = path = Path(c, width=sigma, affine=affine, frame=self.frame)

        return smooth, path

    def predict(self, affine: Quantity) -> path_moments:
        """"""
        return self._path(affine)

    # def fit_predict(
    #     self,
    #     affine,
    #     data: SkyCoord,
    #     timesteps: ndarray,
    #     *,
    #     use_filterpy: Optional[bool] = None,
    #     **kwargs: Any,
    # ):
    #     self.fit(data, timesteps, use_filterpy=use_filterpy, **kwargs)
    #     return self.predict(affine)

    # ===================================================
    # Helper

    @staticmethod
    def make_simple_timesteps(
        ordered_data: SkyCoord, /, dt0: Quantity, *, width: int = 6, vmin: Quantity = 0.01 * u.pc
    ) -> ndarray:
        timesteps = helper.make_timesteps(
            ordered_data,
            dt0=dt0,
            width=width,
            vmin=vmin,
        )
        return timesteps  # TODO! Quantity


if conf.use_filterpy and not HAS_FILTERPY:
    warnings.warn("filterpy not installed. Will use built-in.")
