# -*- coding: utf-8 -*-

"""Kalman Filter code."""

__all__ = ["FirstOrderNewtonianKalmanFilter"]


##############################################################################
# IMPORTS

# STDLIB
from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence, Tuple, Type, Union, cast

# THIRD PARTY
import astropy.units as u
import numpy as np
from astropy.coordinates import BaseCoordinateFrame, BaseRepresentation, CartesianRepresentation
from astropy.coordinates import SkyCoord
from astropy.units import Quantity
from numpy import array, dot, linalg, ndarray
from numpy.lib.recfunctions import structured_to_unstructured

# LOCAL
from . import helper
from trackstream.base import CommonBase
from trackstream.utils.misc import intermix_arrays
from trackstream.utils.path import Path

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
    x0 : (D,) CartesianRepresentation
        Initial position.
    P0 : (2D, 2D) ndarray
        Initial covariance. Must be in the same coordinate representation
        as `x0`.

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
        x0: CartesianRepresentation,  # TODO! as SkyCoord
        P0: ndarray,
        *,
        frame: BaseCoordinateFrame,
        representation_type: Optional[Type[BaseRepresentation]] = None,
        kfv0: Optional[ndarray] = None,
        F0: Union[None, ndarray, Callable[[float], ndarray]] = None,
        Q0: Optional[ndarray] = None,
        R0: ndarray,
        **kwargs: Any,
    ) -> None:
        super().__init__(frame=frame, representation_type=representation_type)
        self._options = kwargs

        ndims = len(x0.components)  # TODO! better setup for 2 dimensions
        if ndims < 2 or ndims > 6:
            raise ValueError

        _x0 = x0.represent_as(self.representation_type)
        _x0 = structured_to_unstructured(_x0._values)

        _kfv0: ndarray = array([0] * ndims) if kfv0 is None else kfv0

        # Initial state
        self._x0 = intermix_arrays(_x0, _kfv0)
        self._P0 = P0  # TODO! make optional
        self._H = helper.make_H(n_dims=ndims)
        self._F0 = F0 if not callable(F0) else None
        self._Q0 = Q0 if not callable(Q0) else None
        self._R0 = R0  # TODO! make function

        self._I = np.eye(2 * ndims)

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
    def process_noise_model(self) -> Callable[..., ndarray]:
        model: Callable = self._process_noise_model
        return model

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
        x = dot(F, x)
        # & covariance matrix
        P = dot(dot(F, P), F.T) + Q

        return x, P

    def _math_update(
        self, z: ndarray, x: ndarray, P: ndarray, R: ndarray, H: ndarray, **kw: Any
    ) -> Tuple[ndarray, ndarray]:
        """Add a new measurement to the Kalman filter.

        Implemented to be compatible with :mod:`filterpy`.

        Parameters
        ----------
        z : (D, 1) ndarray
            Measurement.
        x : (2D, 1) ndarray
            State estimate vector.
        P : (2D, 2D) ndarray
            Covariance matrix.
        R : (D, D) ndarray
            Measurement noise matrix
        H : (2D, 2D) ndarray
            Measurement function.

        **kw : Any
            Absorbs extra arguments for ``filterpy`` compatibility.

        Returns
        -------
        x : (2D, 1) ndarray
            Posterior state estimate.
        P : (2D, 2D) ndarray
            Posterior covariance matrix.
        residual : (2D, 1) ndarray
            Residual between measurement and prediction
        K : (D, D) ndarray
            Kalman gain
        S : (D, D) ndarray
            System uncertainty in measurement space.
        log_lik : (D, D) ndarray
            Log-likelihood. A multivariate normal.
        """
        S = dot(dot(H, P), H.T) + R  # system uncertainty in measurement space
        K = dot(dot(P, H.T), linalg.inv(S))  # Kalman gain

        predict = dot(H, x)  # prediction in measurement space
        residual = z - predict  # measurement and prediction residual

        x = x + dot(K, residual)  # predict new x using Kalman gain

        KH = dot(K, H)
        ImKH = self._I - KH
        # stable representation from Filterpy
        # P = (1 - KH)P(1 - KH)' + KRK'
        P = dot(dot(ImKH, P), ImKH.T) + dot(dot(K, R), K.T)

        # # log-likelihood
        # from scipy.stats import multivariate_normal
        # log_lik = multivariate_normal.logpdf(
        #     z.flatten(),
        #     mean=dot(H, x).flatten(),
        #     cov=S,
        #     allow_singular=True,
        # )

        return x, P  # , residual, K, S, log_lik

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
        P_pred = Ps.copy()

        # initialize parameters
        n, dim_x = Xs.shape
        K = np.zeros((n, dim_x, dim_x))

        # iterate through, running Kalman system again.
        for i in reversed(range(n - 1)):  # [n-2, ..., 0]
            # prediction
            P_pred[i] = dot(dot(Fs[i], P[i]), Fs[i].T) + Qs[i]
            # Kalman gain
            K[i] = dot(dot(P[i], Fs[i].T), linalg.inv(P_pred[i]))

            # update position and covariance
            x_prior = dot(Fs[i], x[i])
            x[i] = x[i] + dot(K[i], (x[i + 1] - x_prior))
            P[i] = P[i] + dot(dot(K[i], P[i + 1] - P_pred[i]), K[i].T)

        return x, P, K, P_pred

    #######################################################
    # Fit

    def fit(
        self,
        data: SkyCoord,
        /,
        timesteps: ndarray,
        representation_type: Optional[Type[BaseRepresentation]] = None,
        **kwargs: Any,
    ) -> Tuple[kalman_output, Path]:
        """Run Kalman Filter with updates on each step.

        Parameters
        ----------
        data : (N,D) ndarray, position-only
        timesteps: (N+1,) ndarray
            Must be start and end-point inclusive.

        **kwargs
            Passed to fit method.

        Returns
        -------
        smooth : `~kalman_output`
            "timesteps", "Xs", "Ps", "Qs"
        """
        q_kw = {**self.options.get("q_kw", {}), **kwargs}  # copy

        dts = np.diff(timesteps)
        if len(dts) != len(data):
            raise ValueError(f"len(timesteps)={len(timesteps)} is not {len(data)+1}")

        # ------ setup ------
        self._original_data = data  # save the data used to fit
        self._timesteps = timesteps  # save the timesteps
        # TODO! check timesteps start with 0

        data = data.transform_to(self.frame).represent_as(self.representation_type)
        Z: ndarray = structured_to_unstructured(data._values)

        # starting points
        x, P = self.x0, self.P0
        R = self.R0  # TODO! make function of data point!

        F = self.state_transition_model(dts[0]) if self.F0 is None else self.F0
        Q = self.process_noise_model(dts[0]) if self.Q0 is None else self.Q0

        n = len(Z)
        # initialize arrays
        Xs = np.empty((n, *np.shape(x)))
        Ps = np.empty((n, *np.shape(P)))
        Fs = np.empty((n, *np.shape(F)))
        Qs = np.empty((n, *np.shape(Q)))

        # iterate predict & update steps
        z: ndarray
        dt: float
        for i, (z, dt) in enumerate(zip(Z, dts)):
            # F, Q
            F = self.state_transition_model(dt)
            Q = self.process_noise_model(dt, **q_kw)

            # predict & update
            x, P = self._math_predict(x=x, P=P, F=F, Q=Q)
            x, P = self._math_update(x=x, P=P, z=z, R=R, H=self._H)

            # append results
            Xs[i], Ps[i] = x, P
            Qs[i] = Q

        # save
        self.__result = kalman_output(timesteps, Xs, Ps, Qs)

        # smoothed
        try:
            xs, Ps, _, Qs = self._math_smoother(Xs, Ps, Fs, Qs)

        except Exception as e:
            print("FIXME!", str(e))
            smooth = self._result
        else:
            smooth = kalman_output(timesteps, xs, Ps, Qs)
            self.__smooth_result = smooth

        # TODO! make sure get the frame and units right
        r = self.representation_type(smooth.Xs[:, ::2].T, unit=u.kpc)
        c = SkyCoord(self.frame.realize_frame(r), copy=False)  # (not interpolated)
        c.representation_type = self.representation_type

        # covariance matrix. select only the phase-space positions
        # everything is Gaussian so there are no off-diagonal elements,
        # so the 1-sigma error is quite easy.
        cov = smooth.Ps[:, ::2, ::2]  # TODO! use H
        var = np.diagonal(cov, axis1=1, axis2=2)
        sigma = np.sqrt(np.sum(np.square(var), axis=-1)) * u.kpc

        # TODO! is this the Affine wanted?
        ci = cast(SkyCoord, c[:-1])
        sp2p = ci.separation(c[1:])  # point-2-point sep
        affine = np.concatenate(([min(Quantity(1e-10, sp2p.unit), 1e-10 * sp2p[0])], sp2p.cumsum()))

        self._path = path = Path(
            c,
            width=sigma,
            affine=affine,
            frame=self.frame,
            representation_type=representation_type,
        )

        print(
            "rep_type",
            path.representation_type,
        )

        return smooth, path

    # def predict(self, affine: Quantity) -
    #     """"""
    #     return self._path(affine)

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
        ordered_data: SkyCoord,
        /,
        dt0: Quantity,
        *,
        width: int = 6,
        vmin: Quantity = Quantity(0.01, u.pc),
    ) -> ndarray:
        onsky = True if cast(u.UnitBase, dt0.unit).physical_type == "angle" else False

        timesteps = helper.make_timesteps(
            ordered_data,
            dt0=dt0,
            width=width,
            vmin=vmin,
            onsky=onsky,
        )
        return timesteps  # TODO! Quantity
