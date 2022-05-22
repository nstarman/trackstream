# -*- coding: utf-8 -*-

"""Kalman Filter code."""

__all__ = ["FirstOrderNewtonianKalmanFilter"]


##############################################################################
# IMPORTS

# STDLIB
from copy import deepcopy
from typing import (
    Any,
    Callable,
    Dict,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

# THIRD PARTY
import astropy.units as u
from astropy.coordinates import (
    BaseCoordinateFrame,
    BaseRepresentation,
    CartesianDifferential,
    CartesianRepresentation,
    SkyCoord,
    UnitSphericalDifferential,
    UnitSphericalRepresentation,
)
from astropy.units import Quantity, StructuredUnit
from astropy.utils.misc import indent
from numpy import (
    arccos,
    arctan2,
    array,
    atleast_2d,
    concatenate,
    convolve,
    cos,
    cumsum,
    diagonal,
    diff,
    dot,
    dtype,
    empty,
    eye,
    insert,
    linalg,
    ndarray,
    ones,
    pi,
    shape,
    sign,
    sin,
    sqrt,
    zeros,
)
from numpy.lib.recfunctions import merge_arrays, structured_to_unstructured
from scipy.linalg import block_diag

# LOCAL
from trackstream._type_hints import CoordinateType
from trackstream.base import CommonBase
from trackstream.track.path import Path, path_moments
from trackstream.utils.misc import intermix_arrays

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


def make_timesteps(data: SkyCoord, *, dt0: Quantity, vmin: Quantity, width: int = 6) -> ndarray:
    """Make distance arrays.

    Parameters
    ----------
    data : (N,) SkyCoord, position-only
        Must be ordered.

    dt0 : Quantity['length'] or Quantity['angle'], keyword-only
        Starting timestep.
    vmin : Quantity['length'] or Quantity['angle'], keyword-only
        Minimum distance, post-convolution.

    width : int,  optional keyword-only
        Number of indices for convolution window. Default is 6.

    Returns
    -------
    timesteps : (N+1,) ndarray
        Smoothed distances, starting with 0.
    """
    onsky = cast(u.UnitBase, dt0.unit).physical_type == "angle"

    # point-to-point distance
    di = cast(SkyCoord, data[1:])
    ds = getattr(di, "separation" if onsky else "separation_3d")(data[:-1])

    # Rolling window
    Ds = convolve(ds, ones((width,)) / width, mode="same")
    # Set minimum
    Ds[Ds < vmin] = vmin

    # munge the starts
    dts = insert(Ds, 0, values=dt0)
    dts = insert(dts, 0, values=0)

    return cumsum(dts.value)  # TODO! as quantity


# TODO check against Q_discrete_white_noise
# from filterpy.common import Q_discrete_white_noise
def make_Q(dt: float, var: float = 1.0, ndims: int = 3, order: int = 2) -> ndarray:
    """Make Q Matrix.

    Parameters
    ----------
    dt : float
    var : float
    ndims : int

    Returns
    -------
    Q : ndarray
    """
    if order == 2:
        # make single-component of q matrix
        q = array(
            [  # single Q matrix
                [0.25 * dt**4, 0.5 * dt**3],  # 1,1 is position
                [0.5 * dt**3, dt**2],  # 2,2 is velocity
            ],
        )
    else:
        raise NotImplementedError

    qs = [q] * ndims  # repeat q for number of dimensions

    Q: ndarray = var * block_diag(*qs)  # block diagonal stack
    return Q


def make_H(ndims: int) -> ndarray:
    """Make H Matrix.

    Parameters
    ----------
    ndims : int

    Returns
    -------
    ndarray
    """
    # component of block diagonal
    h = array([[1, 0], [0, 0]])

    # full matrix is for all components
    # and reduce down to `dim_z` of Kalman Filter, skipping velocity rows
    H: ndarray = block_diag(*([h] * ndims))[::2]

    return H


def make_R(data: ndarray, /) -> ndarray:
    """Make measurement noise covariance matrix from errors.

    Parameters
    ----------
    data : (N, D) ndarray, positional-only
        Rows are the 1-sigma uncorrelated gaussian errors.

    Returns
    -------
    R : (N, D, D) ndarray
        Diagonal array (data.shape[0], data.shape[1], data.shape[1])
        With each diagonal along axis 0 being a row from `data`.

    """
    data = atleast_2d(array(data, copy=False))
    n, dim_x = data.shape

    R = zeros((n, dim_x, dim_x))
    for i in range(dim_x):
        R[:, i, i] = data[:, i] ** 2

    return R


##############################################################################


class FirstOrderNewtonianKalmanFilter(CommonBase):
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

    _onsky: bool
    _kinematics: bool
    _options: Dict[str, Any]

    _units: StructuredUnit
    _x0: ndarray
    _P0: ndarray
    _H: ndarray
    _Q0: Optional[ndarray]

    _process_noise_model: Callable[..., ndarray]

    # results
    __result: Optional[kalman_output]
    __smooth_result: Optional[kalman_output]
    __path: Optional[Path]

    def __init__(
        self,
        x0: BaseRepresentation,
        P0: ndarray,
        *,
        onsky: bool,
        kinematics: bool,
        frame: BaseCoordinateFrame,
        kfv0: Optional[ndarray] = None,
        Q0: Union[None, ndarray, Callable[..., ndarray]] = None,
        **options: Any,
    ) -> None:
        representation_type = UnitSphericalRepresentation if onsky else CartesianRepresentation
        if not kinematics:
            differential_type = None
        elif onsky:  # TODO! need to be cognizant if just radial, etc.
            differential_type = UnitSphericalDifferential
        else:
            differential_type = CartesianDifferential
        super().__init__(
            frame=frame,
            representation_type=representation_type,
            differential_type=differential_type,
        )

        # onsky & kinematics, for dimensionality.
        self._onsky = onsky
        self._kinematics = kinematics

        # Initial position  (requires onsky and kinematics to be set)
        q = self._crd_to_q(x0)
        self._units = cast(StructuredUnit, q.unit)
        _x0 = self._q_to_v(q)

        # Check the number of dimensions
        ndims: int = len(q.dtype.names)
        if ndims < 2 or ndims > 6:
            raise ValueError(f"x0 must have 2 <= x0 <= 6 components, not {ndims}")
        self._I = eye(2 * ndims)  # 2 x dims b/c hidden velocity terms.

        #     KF "velocity". Will be intermixed into `x0`
        _kfv0: ndarray = array([0] * ndims) if kfv0 is None else kfv0
        self._x0 = intermix_arrays(_x0, _kfv0)

        # Kalman Filter
        self._P0 = P0
        self._H = make_H(ndims=ndims)
        self._Q0 = Q0 if not callable(Q0) else None

        # functions
        setattr(self, "_process_noise_model", Q0 if callable(Q0) else make_Q)

        # Running
        self.__result = None
        self.__smooth_result = None
        self.__path = None

        self._options = options  # all the other fitting options.

    # ---------------------------------------------------------------
    # Flags

    @property
    def onsky(self) -> bool:
        """Whether to fit on-sky or 3d."""
        return self._onsky

    @property
    def kinematics(self) -> bool:
        """Whether to fit the kinematics also."""
        return self._kinematics

    @property
    def _dif_attrs(self) -> Tuple[str, ...]:
        attrs = super()._dif_attrs if self.kinematics else ()
        return attrs

    # ---------------------------------------------------------------
    # Initial Conditions

    @property
    def x0(self) -> ndarray:
        """Initial state."""
        return self._x0

    @property
    def P0(self) -> ndarray:
        """Initial state covariance matrix."""
        return self._P0

    @property
    def Q0(self) -> Optional[ndarray]:
        return self._Q0

    @property
    def H(self) -> ndarray:
        return self._H

    @property
    def options(self) -> Dict[str, Any]:
        return self._options

    @staticmethod
    def state_transition_model(dt: float, ndims: int) -> ndarray:
        """Make Transition Matrix.

        Parameters
        ----------
        dt : float
            Time step.

        Returns
        -------
        F : `~numpy.ndarray`
            Block diagonal transition matrix
        """
        # make single block of F matrix
        # [[position to position, position from velocity]
        #  [velocity from position, velocity to velocity]]
        f = array([[1.0, dt], [0, 1.0]])
        # F block-diagonal array
        F: ndarray = block_diag(*([f] * ndims))
        return F

    @property
    def process_noise_model(self) -> Callable[..., ndarray]:
        model: Callable[..., ndarray]
        model = self._process_noise_model
        return model

    # ---------------------------------------------------------------
    # Requires the Kalman filter to be run

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

    @property
    def _path(self) -> Path:
        if self.__path is None:
            raise ValueError(f"need to run {self.__class__.__qualname__}.fit()")
        return self.__path

    # ---------------------------------------------------------------
    # Coordinates <-> internals

    @property
    def units(self) -> StructuredUnit:
        """Units of the coordinates."""
        return self._units

    def _crd_to_q(self, data: Union[CoordinateType, BaseRepresentation], /) -> Quantity:
        """Coordinate-type object to structured Quantity.

        Parameters
        ----------
        data : (N,) SkyCoord or BaseCoordinateFrame or BaseRepresentation, positional-only
            The data to convert to a structured quantity.

        Returns
        -------
        (N, D) Quantity
        """
        # Representation
        if isinstance(data, BaseRepresentation):
            rep = data.represent_as(
                self.representation_type,
                differential_class=self.differential_type if self.kinematics else None,
            )
        else:
            crd = data.transform_to(self.frame)
            rep = crd.represent_as(
                self.representation_type,
                s=self.differential_type if self.kinematics else "base",  # type: ignore
            )
        rv = rep._values
        units = {n: getattr(rep, n).unit for n in rv.dtype.names}
        # TODO! enforce rad if onsky

        # Differentials (optional)
        if self.kinematics:
            dif = rep.differentials["s"]
            dv = dif._values
            # note: merge_arrays adds a dimension to scalar ndarrays
            v = merge_arrays((rv, dv), flatten=True, usemask=False).reshape(rv.shape)
            units.update({n: getattr(dif, n).unit for n in dv.dtype.names})
        else:
            v = rv

        # structured quantity
        su = StructuredUnit(tuple(units.values()), names=tuple(units.keys()))
        q: Quantity = v << su
        return q

    def _q_to_v(self, q: Quantity, /) -> ndarray:
        """Quantity to unstructured array.

        Parameters
        ----------
        q : (N, D) Quantity, positional-only
            Structured Quantity.

        Returns
        -------
        (N, D) ndarray
        """
        # TODO! check units. should match.
        sv: ndarray = q.value
        v: ndarray = structured_to_unstructured(sv)
        return v

    def _crd_to_v(self, crd: Union[CoordinateType, BaseRepresentation], /) -> ndarray:
        """Coordinate / Representation to unstructured array.

        Parameters
        ----------
        crd : (N,) SkyCoord or BaseCoordinateFrame or BaseRepresentation, positional-only
            Coordinates to take an `numpy.ndarray`.

        Returns
        -------
        (N, D) ndarray
        """
        q: Quantity = self._crd_to_q(crd)
        v: ndarray = self._q_to_v(q)
        return v

    def _v_to_q(self, v: ndarray, /) -> Quantity:
        """array to Structured Quantity.

        Parameters
        ----------
        v : (N, D) ndarray, positional-only
            The data to convert to a structured Quantity.

        Returns
        -------
        (N, D) Quantity
        """
        # Make dtype
        dt = dtype([(n, float) for n in (tuple(self._rep_attrs) + tuple(self._dif_attrs))])
        # Make structured quantity
        q: Quantity = v.view(dt) << self.units
        return q

    def _q_to_crd(self, q: Quantity) -> SkyCoord:
        """Quantity to SkyCoord

        Parameters
        ----------
        q : Quantity
            Structured Quantity.

        Returns
        -------
        SkyCoord
        """
        # Differentials
        if not self.kinematics:
            dif = None
        else:
            dif = self.differential_type(**{n: q[n] for n in self._dif_attrs}, copy=False)

        # Representation, including Differentials
        rcs = {n: q[n] for n in self._rep_attrs}
        rep = self.representation_type(**rcs, differentials=dif, copy=False)

        # Frame
        crd: BaseCoordinateFrame = self.frame.realize_frame(
            rep,
            representation_type=self.representation_type,
            differential_type=self.differential_type,
            copy=False,
        )

        return SkyCoord(crd, copy=False)

    def _v_to_crd(self, v: ndarray, /) -> SkyCoord:
        """array to SkyCoord.

        Parameters
        ----------
        v : ndarray
            The data to convert to a SkyCoord

        Returns
        -------
        SkyCoord
        """
        q: Quantity = self._v_to_q(v)
        crd: SkyCoord = self._q_to_crd(q)
        return crd

    # ---------------------------------------------------------------

    def __repr__(self) -> str:
        """String representation."""
        r = ""

        # 1) header (standard repr)
        header: str = object.__repr__(self)
        r += header

        # 2) units
        unit = ", ".join([", ".join((s, s + "/<affine>")) for s in map(str, self.units.values())])
        r += f"\n  units: ({unit})"

        # 3) x0
        r += "\n  x0: " + str(self.x0)

        # 4) P0
        r += "\n  P0: " + indent(str(self.P0), width=6).lstrip()

        # 5) Q0
        r += "\n  Q0: " + indent(str(self.Q0), width=6).lstrip()

        # 6) H
        r += "\n  H : " + indent(str(self.H), width=6).lstrip()

        return r

    #######################################################
    # Math (2 phase + smoothing)

    def _math_predict_and_update(
        self,
        x: ndarray,
        P: ndarray,
        F: ndarray,
        Q: ndarray,
        z: ndarray,
        R: ndarray,
        oz: ndarray,
    ) -> Tuple[ndarray, ndarray]:
        """
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
        oz: (D, 1) ndarray
            Previous measurement.

        Returns
        -------
        x : ndarray
            Prior state vector. The 'predicted' state.
        P : ndarray
            Prior covariance matrix. The 'predicted' covariance.
        """
        # -------------------
        # Prior Prediction

        # if i >= 1006:
        #     import pdb; pdb.set_trace()

        # predict position (not including a control matrix)
        x = dot(F, x)

        # & covariance matrix
        P = dot(dot(F, P), F.T) + Q

        # -------------------
        # Posterior

        S = dot(dot(self._H, P), self._H.T) + R  # system uncertainty in measurement space
        K = dot(dot(P, self._H.T), linalg.inv(S))  # Kalman gain

        predict = dot(self._H, x)  # prediction in measurement space
        residual = z - predict  # measurement and prediction residual

        if self.onsky:  # unwrap to keep x close to z
            # first coordinate is always the longitude
            deltalon = z[0] - x[0]
            pa = arctan2(sin(deltalon), 0)  # position angle
            residual[0] = sign(pa) * arccos(cos(deltalon))

            # TODO! similar for |Latitude|

        x = x + dot(K, residual)  # predict new x using Kalman gain

        if self.onsky:  # need to correct for phase-wrap
            # first coordinate is always the longitude
            # keeps in (-180, 180) deg
            x[0] += 2 * pi if (x[0] < -pi) else 0
            x[0] -= 2 * pi if (x[0] >= pi) else 0
            # todo! similar unwrapping of the |Latitude|

        KH = dot(K, self._H)
        ImKH = self._I - KH
        # stable representation using Joseph equation (from Filterpy)
        # P = (1 - KH)P(1 - KH)' + KRK'
        P = dot(dot(ImKH, P), ImKH.T) + dot(dot(K, R), K.T)

        return x, P

    def _rts_smoother(
        self,
        dts: ndarray,
        Xs: ndarray,
        Ps: ndarray,
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
        x = deepcopy(Xs)
        P = deepcopy(Ps)
        P_pred = deepcopy(Ps)
        Qs = deepcopy(Qs)

        # Initialize parameters
        n, dims_x = Xs.shape
        ndims = dims_x // 2
        K = zeros((n, dims_x, dims_x))

        # Iterate through, running Kalman system again.
        for i in reversed(range(n - 1)):  # [n-2, ..., 0]
            F = self.state_transition_model(dts[i], ndims=ndims)
            # TODO? have F as a pre-computed array

            # prediction
            P_pred[i] = dot(dot(F, P[i]), F.T) + Qs[i]
            # Kalman gain
            K[i] = dot(dot(P[i], F.T), linalg.inv(P_pred[i]))

            # update position and covariance
            x_prior = dot(F, x[i])
            x[i] = x[i] + dot(K[i], (x[i + 1] - x_prior))
            P[i] = P[i] + dot(dot(K[i], P[i + 1] - P_pred[i]), K[i].T)

            if self.onsky:  # need to correct for phase-wrap
                # first coordinate is always the longitude
                # keeps in (-180, 180) deg
                x[i][0] += 2 * pi if (x[i][0] < -pi) else 0
                x[i][0] -= 2 * pi if (x[i][0] >= pi) else 0

        return x, P, K, P_pred

    #######################################################
    # Fit

    def fit(
        self, data: SkyCoord, errors: Union[ndarray, Quantity], /, timesteps: ndarray, **kwargs: Any
    ) -> Tuple[Path, kalman_output]:
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
        # ------ setup ------

        # Get the time deltas from the time steps.
        # Checking the time steps are compatible with the data.
        if timesteps[0] != 0:
            raise ValueError(f"timesteps should start with 0, not {timesteps[0]}")
        dts = diff(timesteps)
        if len(dts) != len(data):
            raise ValueError(f"len(timesteps)={len(timesteps)} is not {len(data)+1}")
        elif any(dts < 0):
            raise ValueError("timesteps must be >= 0")

        path_name = kwargs.pop("name", None)

        # Get Z, in correct units.
        Zs = self._crd_to_v(data)
        num_z = len(Zs)

        # TODO! it would be great to be able to transform errors
        # as well
        Rs = errors

        # ------ IC (i-1) ------

        # KF
        x, P = self.x0, self.P0
        ndims = len(x) // 2
        Q = self.process_noise_model(dts[0], ndims=ndims) if self.Q0 is None else self.Q0  # FIXME!

        # Data
        oz = self.x0  # i-1 'data' is KF starting point

        # initialize arrays
        Xs = empty((num_z, *shape(x)))
        Ps = empty((num_z, *shape(P)))
        Qs = empty((num_z, *shape(Q)))

        q_kw = {**self.options.get("q_kw", {}), **kwargs}  # copy

        # ------ run ------
        # iterate predict & update steps
        z: ndarray
        dt: float
        for i, (z, R, dt) in enumerate(zip(Zs, Rs, dts)):
            # F_(i-1, i)
            F = self.state_transition_model(dt, ndims=ndims)
            Q = self.process_noise_model(dt, ndims=ndims, **q_kw)

            # predict & update
            x, P = self._math_predict_and_update(x=x, P=P, F=F, Q=Q, z=z, R=R, oz=oz)

            # append results
            Xs[i], Ps[i] = x, P
            Qs[i] = Q

        # save
        self.__result = kalman_output(timesteps, Xs, Ps, Qs)

        # smoothed
        try:
            xs, Ps, _, Qs = self._rts_smoother(dts, Xs, Ps, Qs)
        except Exception as e:
            print("can't smooth", str(e))
            self.__smooth_result = smooth = self.__result
        else:
            smooth = kalman_output(timesteps, xs, Ps, Qs)
            self.__smooth_result = smooth

        # ------ make path ------

        # Measured Xs
        # mXs = dot(self._H, smooth.Xs.T).T  # (N, D)
        mc = self._v_to_crd(array(smooth.Xs[:, ::2], copy=True))  # needs to be C-continuous

        # Covariance matrix.
        # We only care about the real coordinates, not the KF `velocity'.
        # The block elements are uncorrelated.  # TODO! more general
        cov = smooth.Ps[:, ::2, ::2]
        var = diagonal(cov, axis1=1, axis2=2)
        width = self._v_to_q(sqrt(var))

        # Affine  # TODO! is this the Affine wanted?
        ci = cast(SkyCoord, mc[:-1])
        cj = cast(SkyCoord, mc[1:])
        sp2p = ci.separation(cj)  # point-2-point sep
        minafn = min(Quantity(1e-10, sp2p.unit), 1e-10 * sp2p[0])
        affine = concatenate((array(minafn, subok=True), sp2p.cumsum()))

        self.__path = path = Path(
            mc,
            width=width,
            amplitude=None,  # TODO!
            # keyword-only
            name=path_name,
            affine=affine,
            frame=self.frame,
            representation_type=self.representation_type,
            meta=None,
        )

        return path, smooth

    def predict(self, affine: Quantity, /) -> path_moments:
        """Predict the Kalman Filter path.

        Parameters
        ----------
        affine: Quantity

        Returns
        -------
        SkyCoord

        Raises
        ------
        ValueError
            If the KF is not ``fit()``.
        """
        return self._path(affine)

    def fit_predict(
        self,
        affine: Quantity,
        data: SkyCoord,
        errors: Union[ndarray, Quantity],
        /,
        timesteps: ndarray,
        **kwargs: Any,
    ) -> path_moments:
        self.fit(data, errors, timesteps=timesteps, **kwargs)
        return self.predict(affine)
