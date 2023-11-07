"""Kalman Filter code."""


from __future__ import annotations

from copy import deepcopy
from dataclasses import KW_ONLY, dataclass
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, cast, final
import warnings

import astropy.units as u
from numpy import dot
import numpy as np
import numpy.lib.recfunctions as rfn
from numpy.linalg import inv
from scipy.linalg import block_diag

from trackstream._typing import NDFloating  # noqa: TCH001, RUF100
from trackstream.stream.core import StreamArm
from trackstream.track.fit.exceptions import EXCEPT_NO_KINEMATICS
from trackstream.track.fit.kalman.utils import intermix_arrays
from trackstream.track.fit.utils import FrameInfo
from trackstream.track.utils import is_structured
from trackstream.utils.coord_utils import f2q
from trackstream.utils.unit_utils import merge_units

if TYPE_CHECKING:
    from astropy.units import Quantity
    from typing_extensions import Self

    from trackstream.track.width.plural import Widths

__all__: list[str] = []

##############################################################################
# PARAMETERS


class kalman_output(NamedTuple):
    """Kalman Filter output."""

    timesteps: NDFloating
    x: NDFloating
    P: NDFloating


@final
@dataclass(frozen=True)
class KFInfo(FrameInfo):
    """Kalman Filter information."""

    REGISTRY: ClassVar[dict[type, KFInfo]] = {}


##############################################################################
# CODE
##############################################################################


@dataclass(frozen=True)
class X0Field:
    """Dataclass for x0 field."""

    def __set_name__(self, owner: type, name: str) -> None:
        self.name: str
        object.__setattr__(self, "name", "_" + name)

    def __get__(self, instance: FONKFBase, owner: type) -> NDFloating:
        if instance is None:
            raise AttributeError
        return getattr(instance, self.name)

    def __set__(self, instance: FONKFBase, value: NDFloating) -> None:
        if len(value.shape) != 1:
            msg = "x0 must be 1D"
            raise ValueError(msg)
        if len(value) % 2 != 0:
            msg = "x0 must have an even number of dimensions."
            raise ValueError(msg)

        nd = len(value) // 2
        if nd < 2 or nd > 6:
            msg = f"x0 must have 2 <= x0 <= 6 components, not {nd}"
            raise ValueError(msg)

        object.__setattr__(instance, self.name, value)


@dataclass(frozen=True)
class P0Field:
    """Dataclass for P0 field."""

    def __set_name__(self, owner: type, name: str) -> None:
        self.name: str
        object.__setattr__(self, "name", "_" + name)

    def __get__(self, instance: FONKFBase, owner: type) -> NDFloating:
        if instance is None:
            raise AttributeError
        return getattr(instance, self.name)

    def __set__(self, instance: FONKFBase, value: NDFloating) -> None:
        if len(value.shape) != 2:
            msg = "P0 must be 2D"
            raise ValueError(msg)
        if not np.all(np.array(value.shape) % 2 == 0):
            msg = "P0 must have an even number of dimensions."
            raise ValueError(msg)

        object.__setattr__(instance, self.name, value)


@dataclass(frozen=True)
class FONKFBase:
    """First-order Newtonian Kalman filter class.

    Parameters
    ----------
    x0 : (D,) BaseRepresentation
        Initial position.
    P0 : (2D, 2D) ndarray
        Initial covariance. Must be in the same coordinate representation
        as `x0`.

    frame : BaseCoordinateFrame, keyword-only
        The frame of the data, without
    kfv0 : ndarray
        Initial filter 'velocity'.

    Q0 : ndarray callable or None, optional keyword-only
    R0 : ndarray callable or None, optional keyword-only
    """

    x0: X0Field = X0Field()
    P0: P0Field = P0Field()
    _: KW_ONLY
    Q0: NDFloating | None
    info: ClassVar[KFInfo]

    def __post_init__(self) -> None:
        # H
        self.H: NDFloating
        # component of block diagonal
        h = np.array([[1, 0], [0, 0]])
        # full matrix is for all components
        # and reduce down to `dim_z` of Kalman Filter, skipping velocity rows
        H: NDFloating = block_diag(*([h] * self.nfeature))[::2]
        object.__setattr__(self, "H", H)

        # I
        self._I: NDFloating
        object.__setattr__(self, "_I", np.eye(2 * self.nfeature))

    # ===============================================================

    @singledispatchmethod
    @classmethod
    def from_format(
        cls,
        arm: object,  # noqa: ARG003
        *,
        kinematics: bool | None = None,  # noqa: ARG003
        width0: None | Widths = None,  # noqa: ARG003
    ) -> Any:  # https://github.com/python/mypy/issues/11727
        """Create a Kalman Filter from an object.

        Parameters
        ----------
        arm : object, positional-only
            The object to create the Kalman Filter from.
        kinematics : bool | None, optional
            Whether to use kinematics. If `None`, the data will be checked.
        width0 : None | Widths, optional
            The initial widths to use.

        Returns
        -------
        `~trackstream.track.fit.kalman.base.FONKFBase`
            The Kalman Filter.
        """
        msg = "not dispatched"
        raise NotImplementedError(msg)

    @from_format.register(StreamArm)
    @classmethod
    def _from_format_streamarm(  # noqa: C901
        cls: type[Self],
        arm: StreamArm,
        *,
        kinematics: bool | None = None,
        width0: None | u.Quantity = None,
    ) -> Self:
        """Make Kalman Filter from a stream.

        Parameters
        ----------
        arm : `trackstream.stream.StreamArm`
            The stream (arm) from which to build the Kalman filter

        kinematics : bool | None, optional keyword-only
            Whether to use kinematics.
        width0 : None | `astropy.units.Quantity`, optional keyword-only
            The initial widths to use.

        Returns
        -------
        `trackstream.fit.kalman.FONKFBase`
        """
        # flags
        if kinematics is None:
            kinematics = arm.has_kinematics
        elif kinematics is True and not arm.has_kinematics:
            raise EXCEPT_NO_KINEMATICS

        if arm.frame is None:
            # LOCAL
            from trackstream.stream.base import FRAME_NONE_ERR

            raise FRAME_NONE_ERR

        # Setup
        info = cls.info  # all necessary info to extract data
        nfeature = len(info.components(kinematics=kinematics))  # index for slicing

        f = arm.coords[0]
        f.representation_type = info.representation_type
        f.differential_type = info.differential_type
        dtype_names = f2q(f, flatten=True).dtype.names

        orgn = arm.origin.transform_to(arm.frame)
        orgn.representation_type = info.representation_type
        orgn.differential_type = info.differential_type
        ov = rfn.structured_to_unstructured(f2q(orgn).to_value(info.units))[:nfeature]

        # -------------------
        # Initial Conditions
        # now that everything is in the right frame, we just need to construct
        # the matrices.
        #
        # Starting Position
        # starting from the origin. It might be better to start from a locus of
        # points on the stream, near the origin. Or the Lagrange point. But we
        # don't have that info.
        # Corresponding kalman `velocity`
        kfv0 = np.array([0] * nfeature)
        # Initial conditions
        x0 = intermix_arrays(ov, kfv0)

        # Initial Uncertainty and Stream Width
        # This is less important to get exactly correct.
        flat_units = merge_units(info.units)

        if width0 is None:
            width0 = u.Quantity(np.zeros((), dtype=[(n, float) for n in flat_units.field_names]), flat_units)
        elif not isinstance(width0, u.Quantity) or not is_structured(width0):
            raise ValueError

        ps: list[NDFloating] = []
        for rn, fn in zip(info.components(kinematics=kinematics), dtype_names, strict=False):
            # ^ relying on zip-shortest to cut off svs iter b/c that always
            # includes the kinematics.
            unit = flat_units[rn]
            if rn in width0.dtype.names:
                wn0 = cast("Quantity", width0[rn]).to_value(unit).item()
            elif fn in width0.dtype.names:
                wn0 = cast("Quantity", width0[fn]).to_value(unit).item()
            else:
                msg = f"{rn} & {fn} are not in the stream data, setting the width to 0."
                warnings.warn(msg, stacklevel=2)
                wn0 = 0

            # The R contribution to the error
            # there are 2 options, the frame or the rep component name.
            if (fne := f"{fn}_err") in arm.data.columns:
                rn0 = arm.data[fne][:3].mean().to_value(unit)
            elif (rne := f"{rn}_err") in arm.data.columns:
                rn0 = arm.data[rne][:3].mean().to_value(unit)
            else:
                msg = f"{fne} & {rne} are not in the stream data, setting the error to the width."
                warnings.warn(msg, stacklevel=2)
                rn0 = 0

            # combine data error with stream width
            pn = rn0**2 + wn0**2

            # covariance block
            p = np.array([[pn, 0], [0, pn]])

            ps.append(p)

        P0 = block_diag(*ps)  # Covariance matrix
        Q0 = np.zeros_like(P0)  # Process Noise Model

        return cls(x0=x0, P0=P0, Q0=Q0)

    # ---------------------------------------------------------------

    @property
    def nfeature(self) -> int:
        """Total number of dimensions of the Kalman Filter."""
        return len(self.x0) // 2

    # ---------------------------------------------------------------
    # Initial Conditions

    def state_transition_model(self, dt: NDFloating) -> NDFloating:
        """Make Transition Matrix.

        Parameters
        ----------
        dt : (N,) or (N, 2) array
            Time step or array thereof. Must be a structured ndarray with fields
            ``positions`` and ``kinematics`` (if
            `~trackstream.fit.kalman.FONKFBase.kinematics` is `True`).

        Returns
        -------
        F : ndarray
            Block diagonal transition matrix. The shape will be (1, nfeature,
            nfeature) or (N, nfeature, nfeature), depending if ``dt`` was scalar or of
            length ``N``.
        """
        if len(dt.shape) == 1 or dt.shape[1] == 1:
            dt = np.c_[dt, dt]
        elif len(dt.shape) > 2:
            raise ValueError

        # # make single block of F matrix
        # # F block-diagonal array
        nd = self.nfeature
        F = np.zeros((len(dt), 2 * nd, 2 * nd))
        idx = np.arange(2 * nd, dtype=int)[::2]

        F[:, idx, idx] = 1  # positions
        F[:, idx + 1, idx + 1] = 1  # kf `velocities'
        # off diagonals
        ik = nd // 2  # assuming same number off q & p
        F[:, idx[:ik], idx[:ik] + 1] = dt[:, 0][:, None]
        F[:, idx[ik:], idx[ik:] + 1] = dt[:, 1][:, None]

        return F

    def process_noise_model(self, dt: NDFloating, var: float = 1.0) -> NDFloating:
        """Process noise.

        Parameters
        ----------
        dt : (N, K) ndarray
            N data points of K types (1 if positions, 2 if also kinematics)
        var : float
            Variance of the process noise.

        Returns
        -------
        (D, D) ndarray
            The ``Q`` term of a Kalman filter.
        """
        if len(dt.shape) == 1 or dt.shape[1] == 1:
            dt = np.c_[dt, dt]
        elif len(dt.shape) > 2:
            raise ValueError

        nd = self.nfeature
        Q = np.zeros((np.shape(dt)[0], 2 * nd, 2 * nd))

        # Fill in block-diagonal
        # single-component of Q matrix
        #     [[0.25 * dt**4, 0.5 * dt**3],  # 1,1 is position
        #      [0.5 * dt**3, dt**2]]  # 2,2 is kf-`velocity'
        idx = np.arange(2 * nd, dtype=int)[::2]
        ik = nd // 2
        Q[:, idx[:ik], idx[:ik]] = 0.25 * dt[:, 0][:, None] ** 4
        Q[:, idx[:ik], idx[:ik] + 1] = 0.5 * dt[:, 0][:, None] ** 3
        Q[:, idx[:ik] + 1, idx[:ik]] = 0.5 * dt[:, 0][:, None] ** 3
        Q[:, idx[:ik] + 1, idx[:ik] + 1] = dt[:, 0][:, None] ** 2

        Q[:, idx[ik:], idx[ik:]] = 0.25 * dt[:, 1][:, None] ** 4
        Q[:, idx[ik:], idx[ik:] + 1] = 0.5 * dt[:, 1][:, None] ** 3
        Q[:, idx[ik:] + 1, idx[ik:]] = 0.5 * dt[:, 1][:, None] ** 3
        Q[:, idx[ik:] + 1, idx[ik:] + 1] = dt[:, 1][:, None] ** 2

        return var * Q

    # ---------------------------------------------------------------
    # Hooks for subclasses.

    def _wrap_residual(self, x: NDFloating) -> NDFloating:
        return x

    def _wrap_posterior(self, x: NDFloating) -> NDFloating:
        return x

    #######################################################
    # Math (2 phase + smoothing)

    def _math_predict_and_update(  # noqa: PLR0913
        self,
        x: NDFloating,
        P: NDFloating,
        F: NDFloating,
        Q: NDFloating,
        z: NDFloating,
        R: NDFloating,
    ) -> tuple[NDFloating, NDFloating]:
        """Predict and update step.

        Predict prior using Kalman filter transition functions.
        Then add a new measurement to the Kalman filter.

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
        z : (D, 1) ndarray
            Measurement.
        R : (D, D) ndarray
            Measurement noise matrix.

        Returns
        -------
        x : ndarray
            Prior state vector. The 'predicted' state.
        P : ndarray
            Prior covariance matrix. The 'predicted' covariance.
        """
        # -------------------
        # Prior Prediction

        # predict position (not including a control matrix)
        x = dot(F, x)

        # & covariance matrix
        P = dot(dot(F, P), F.T) + Q

        # -------------------
        # Posterior

        S = dot(dot(self.H, P), self.H.T) + R  # system uncertainty in measurement space
        K = dot(dot(P, self.H.T), inv(S))  # Kalman gain

        predict = dot(self.H, x)  # prediction in measurement space
        residual = z - predict  # measurement and prediction residual
        residual = self._wrap_residual(residual)

        x = x + dot(K, residual)  # predict new x using Kalman gain
        x = self._wrap_posterior(x)

        KH = dot(K, self.H)
        ImKH = self._I - KH
        # stable representation using Joseph equation (from Filterpy)
        # P = (1 - KH)P(1 - KH)' + KRK'
        P = dot(dot(ImKH, P), ImKH.T) + dot(dot(K, R), K.T)

        return x, P

    def _rts_smoother(
        self,
        Xs: NDFloating,
        Ps: NDFloating,
        Fs: NDFloating,
        Qs: NDFloating,
    ) -> tuple[NDFloating, NDFloating]:
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
        """
        # copy
        x = deepcopy(Xs)
        P = deepcopy(Ps)
        Pp = deepcopy(Ps)

        # Initialize parameters
        n, dims_x = Xs.shape
        K = np.zeros((n, dims_x, dims_x))

        # Iterate through, running Kalman system again.
        # for i in reversed(range(n - 1)):  # [n-2, ..., 0]
        for k in range(n - 2, -1, -1):
            F = Fs[k]

            # prediction
            Pp[k] = dot(dot(F, P[k]), F.T) + Qs[k]
            # Kalman gain
            K[k] = dot(dot(P[k], F.T), inv(Pp[k]))

            # update position and covariance
            x_prior = dot(F, x[k])
            residual = x[k + 1] - x_prior
            residual = self._wrap_residual(residual)

            x[k] += dot(K[k], residual)
            P[k] += dot(dot(K[k], P[k + 1] - Pp[k]), K[k].T)

            x[k][:] = self._wrap_posterior(x[k])

        return x, P

    #######################################################
    # Fit

    def fit(
        self,
        data: NDFloating,
        /,
        errors: NDFloating,
        widths: NDFloating,
        timesteps: NDFloating,
    ) -> tuple[kalman_output, kalman_output]:
        """Run Kalman Filter with updates on each step.

        Parameters
        ----------
        self : ``FONKFBase``
            The Kalman filter to run.
        data : (N,) SkyCoord, positional-only
            The data to fit with the Kalman filter.

        errors : (N,) Quantity
            The error on the data.
        widths : (N,) Quantity
            The width of the data.
        timesteps: (N,) or (N, 2) ndarray
            Must be start and end-point inclusive.

        Returns
        -------
        path : `~trackstream.fit.path.Path`
        """
        # ------ setup ------

        N = len(data)

        # Get the time deltas from the time steps.
        # Checking the time steps are compatible with the data.
        error, msg = False, ""
        if len(timesteps) != N:
            msg = f"len(timesteps)={len(timesteps)} is not {N}"
        elif len(widths) != N:
            msg = f"len(widths)={len(widths)} is not {N}"
        elif len(errors) != N:
            msg = f"len(errors)={len(errors)} is not {N}"
        elif np.any(timesteps < 0):
            msg = "timesteps must be >= 0"

        if error:
            raise ValueError(msg)

        # Widths
        Ws = np.zeros((N, self.nfeature))
        Ws[:] = widths**2

        # Error matrix
        #
        # TODO: it would be great to be able to transform errors as well. For
        # now, the errors must be in kalman filter's rep/diff type and units
        idx = np.arange(self.nfeature)  # diagonal indices
        Rs = np.zeros((N, self.nfeature, self.nfeature))
        Rs[:, idx, idx] = errors**2  # assign to diagonal

        # ------ IC (i-1) ------

        x, P = self.x0, self.P0  # KF

        # initialize arrays
        Xs = np.empty((N, *np.shape(x)))
        Ps = np.empty((N, *np.shape(P)))

        # Make the transition model and process noise model
        Fs = self.state_transition_model(timesteps)
        Qs = self.process_noise_model(timesteps, var=1)

        # ------ run ------
        # iterate predict & update steps
        z: NDFloating
        for i, (z, R, F, Q) in enumerate(zip(data, Rs, Fs, Qs, strict=True)):
            R[idx, idx] += Ws[i]  # add stream width to uncertainty
            # TODO: this is at the previous step! need to
            # do it during predict / update, not before
            # predict & update
            x, P = self._math_predict_and_update(x=x, P=P, F=F, Q=Q, z=z, R=R)

            # append results
            Xs[i], Ps[i] = x, P

        # results
        result = kalman_output(timesteps, Xs, Ps)

        # smoothed
        Xs, Ps = self._rts_smoother(Xs, Ps, Fs, Qs)
        smooth = kalman_output(timesteps, Xs, Ps)

        return result, smooth
