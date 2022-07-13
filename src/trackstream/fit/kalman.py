"""Kalman Filter code."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import warnings
from collections.abc import Mapping
from copy import deepcopy
from typing import TYPE_CHECKING, Any, NamedTuple, ValuesView, cast

# THIRD PARTY
import astropy.units as u
import numpy as np
from astropy.coordinates import (
    BaseCoordinateFrame,
    BaseDifferential,
    BaseRepresentation,
    CartesianDifferential,
    CartesianRepresentation,
    SkyCoord,
    UnitSphericalDifferential,
    UnitSphericalRepresentation,
)
from astropy.table import QTable
from astropy.units import Quantity, StructuredUnit
from attrs import converters, define, field
from numpy import arccos, arctan2, cos, dot, empty, ndarray, sign, sin
from numpy.lib.recfunctions import (
    apply_along_fields,
    merge_arrays,
    structured_to_unstructured,
    unstructured_to_structured,
)
from numpy.linalg import inv
from scipy.linalg import block_diag
from typing_extensions import Self

# LOCAL
from .errors import EXCEPT_3D_NO_DISTANCES, EXCEPT_NO_KINEMATICS
from trackstream._typing import CoordinateType
from trackstream.base import FramedBase
from trackstream.fit.track.path import Path
from trackstream.utils.coord_utils import resolve_framelike
from trackstream.utils.misc import intermix_arrays, is_structured

if TYPE_CHECKING:
    # LOCAL
    from trackstream.stream import StreamArm


__all__ = ["FirstOrderNewtonianKalmanFilter"]

##############################################################################
# PARAMETERS


class kalman_output(NamedTuple):
    timesteps: ndarray
    x: ndarray
    P: ndarray


def is_negative(x, axis=None):
    return x < 0


##############################################################################
# CODE
##############################################################################


def make_error(kf: FirstOrderNewtonianKalmanFilter, error_table: QTable, default: float = 0) -> Quantity:
    """Get Error.

    Parameters
    ----------
    data : `~astropy.table.QTable`
        The data table from which to extract the error information.
    kf : FirstOrderNewtonianKalmanFilter
        The Kalman filter to use to understand what information should be gotten
        from ``data``.
    default : float, optional
        Default error value, by default 0.

    Returns
    -------
    (N, D) ndarray
    """
    # Get errrors in each component
    errors = np.empty(len(error_table), dtype=[(n, float) for n in kf.units.field_names])
    for fn, rn in kf.frame_component_names.items():
        unit = kf.units[rn]

        if (fne := f"{fn}_err") in error_table.columns:
            r = error_table[fne].to_value(unit)
        elif (rne := f"{rn}_err") in error_table.columns:
            r = error_table[fne].to_value(unit)
        else:
            msg = f"{fne} and {rne} are not in the data; setting to the default."
            warnings.warn(msg)
            r = u.Quantity(default, unit).to_value(unit)

        errors[rn] = r**2  # it work on the variance

    return errors << kf.units


def _rep_to_array(rep: BaseRepresentation, /) -> tuple[ndarray, u.StructuredUnit]:
    """Representation to Quantity.

    Parameters
    ----------
    rep : (N,) BaseRepresentation
        The representation data.

    Returns
    -------
    sv : (N,) ndarray
        Structure value.
    `~astropy.units.StructuredUnit`
        Corresponding units.
    """
    rv = rep._values  # structured ndarray
    units = rep._units

    # Differentials (optional)
    if "s" not in rep.differentials:
        v = rv
    else:
        dif = rep.differentials["s"]
        dv = dif._values
        # note: merge_arrays adds a dimension to scalar ndarrays
        v = merge_arrays((rv, dv), flatten=True, usemask=False).reshape(rv.shape)
        units.update(dif._units)

    # structured quantity
    su = StructuredUnit(tuple(units.values()), names=tuple(units.keys()))
    return v, su


##############################################################################


@define(frozen=True)
class FirstOrderNewtonianKalmanFilter(FramedBase):
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

    x0: ndarray = field(converter=np.array)
    P0: ndarray = field(converter=np.array)
    Q0: ndarray | None = field(default=None, converter=converters.optional(np.array))
    H: ndarray = field(init=False, repr=False)  # always default

    onsky: bool = field(kw_only=True)
    kinematics: bool = field(kw_only=True)
    options: dict[str, Any] = field(factory=dict, kw_only=True)

    frame: BaseCoordinateFrame = field(kw_only=True, converter=resolve_framelike)
    """The frame of this object, set at initialization."""
    frame_representation_type: type[BaseRepresentation] = field(kw_only=True)
    frame_differential_type: type[BaseDifferential] | None = field(kw_only=True)
    units: StructuredUnit = field(default=None, kw_only=True)  # set by default
    _wrap_at: ndarray = field(init=False, repr=False, eq=False)
    _angle_convert: ndarray = field(init=False, repr=False, eq=False)

    _I: ndarray = field(init=False, repr=False, eq=False)

    @H.default  # type: ignore
    def _H_default(self):
        # component of block diagonal
        h = np.array([[1, 0], [0, 0]])
        # full matrix is for all components
        # and reduce down to `dim_z` of Kalman Filter, skipping velocity rows
        H: ndarray = block_diag(*([h] * self.ndims))[::2]
        return H

    @_I.default  # type: ignore
    def _I_default(self) -> ndarray:
        return np.eye(2 * self.ndims)

    @frame_representation_type.default  # type: ignore
    def _representation_type_default(self) -> type[BaseRepresentation]:
        return UnitSphericalRepresentation if self.onsky else CartesianRepresentation

    @frame_differential_type.default  # type: ignore
    def _differential_type_default(self) -> type[BaseDifferential] | None:
        if not self.kinematics:
            dif_type = None
        elif self.onsky:  # TODO! need to be cognizant if just radial, etc.
            dif_type = UnitSphericalDifferential
        else:
            dif_type = CartesianDifferential

        return dif_type

    @_wrap_at.default  # type: ignore
    def _wrap_at_default(self) -> None:
        if self.onsky:
            uns = tuple(self.units.values())
            lon = Quantity(180, u.deg).to_value(uns[0])
            lat = Quantity(90, u.deg).to_value(uns[1])
            wrap_at = np.array([lon, lat])
        else:
            wrap_at = np.array([])
        return wrap_at

    @_angle_convert.default  # type: ignore
    def _angle_convert_default(self) -> None:
        if self.onsky:
            uns = tuple(self.units.values())
            lon = uns[0].to(u.rad)
            lat = uns[1].to(u.rad)
            convert = np.array([lon, lat])
        else:
            convert = np.array([1, 1])
        return convert

    @x0.validator  # type: ignore
    def _x0_validate(self, _, value: ndarray) -> None:
        if len(value.shape) != 1:
            raise ValueError("x0 must be 1D")
        elif len(value) % 2 != 0:
            raise ValueError("x0 must have an even number of dimensions.")

        nd = self.ndims
        if nd < 2 or nd > 6:
            raise ValueError(f"x0 must have 2 <= x0 <= 6 components, not {nd}")

    @P0.validator  # type: ignore
    def _P0_validate(self, _, value: ndarray) -> None:
        if len(value.shape) != 2:
            raise ValueError("P0 must be 2D")
        if not np.all(np.array(value.shape) % 2 == 0):
            raise ValueError("P0 must have an even number of dimensions.")

    # -----------------------------------------------

    @classmethod
    def from_stream(
        cls: type[Self],
        arm: StreamArm,
        /,
        onsky: bool | None = None,
        kinematics: bool | None = None,
        *,
        width0: None | Quantity | dict[str, Quantity] = None,
        **options: Any,
    ) -> Self:
        """Make Kalman Filter from a stream.

        Parameters
        ----------
        arm : `trackstream.stream.StreamArm`
            The stream (arm) from which to build the Kalman filter

        Returns
        -------
        `trackstream.fit.kalman.FirstOrderNewtonianKalmanFilter`
        """
        # flags
        if onsky is None:
            onsky = not arm.has_distances
        elif onsky is False and not arm.has_distances:
            raise EXCEPT_3D_NO_DISTANCES
        if kinematics is None:
            kinematics = arm.has_kinematics
        elif kinematics is True and not arm.has_distances:
            raise EXCEPT_NO_KINEMATICS

        # Representation
        rep_type = UnitSphericalRepresentation if onsky else CartesianRepresentation
        rac = getattr(rep_type, "attr_classes", ())
        nrepdims: int = len(rac)

        # Kinematics
        dif_type: type[BaseDifferential] | None
        ndifdims: int
        if not kinematics:
            dif_type = None  # strip kinematics
            ndifdims = 0
        elif onsky:
            # TODO! need to be cognizant if just radial, etc.
            dif_type = UnitSphericalDifferential
            dac = getattr(dif_type, "attr_classes", ())
            ndifdims = len(dac)
        else:
            dif_type = CartesianDifferential
            dac = getattr(dif_type, "attr_classes", ())
            ndifdims = len(dac)

        # Total number of dimensions
        ndims = nrepdims + ndifdims

        # Build and convert to correct frame & rep/diff-type
        frame = arm.coords.frame.replicate_without_data(representation_type=rep_type, differential_type=dif_type)
        rep = arm.coords.transform_to(frame).represent_as(rep_type, s=dif_type, in_frame_units=True)

        # Get component names and units
        component_names = frame.get_representation_component_names()
        component_units = {k: getattr(rep, k).unit for k in component_names.values()}
        if kinematics:
            dif = rep.differentials["s"]
            dif_names = frame.get_representation_component_names("s")
            component_names.update(dif_names)
            component_units.update({k: getattr(dif, k).unit for k in dif_names.values()})

        units = StructuredUnit(tuple(component_units.values()), names=tuple(component_names.values()))

        # -------------------
        # Initial Conditions
        # now that everything is in the right frame, we just need to construct
        # the matrices.

        # Starting Position
        # starting from the origin. It might be better to start from a locus of
        # points on the stream, near the origin. Or the Lagrange point. But we
        # don't have that info.
        orep = arm.origin.transform_to(frame).represent_as(rep_type, s=dif_type, in_frame_units=True)
        sdata, su = _rep_to_array(orep)  # structure ndarray & units
        rdata = structured_to_unstructured(sdata) * structured_to_unstructured(su.to(units))
        x0q = rdata  # .mean(axis=0)
        # Corresponding kalman velocity
        kfv0: ndarray = np.array([0] * ndims)
        # Initial conditions
        x0 = intermix_arrays(x0q, kfv0)

        # Initial Uncertainty and Stream Width
        # This is less important to get exactly correct.
        if width0 is None:
            width0 = {}
        elif isinstance(width0, Mapping):
            pass
        elif is_structured(width0):
            width0 = {k: cast(Quantity, width0[k]) for k in width0.dtype.names}
        else:
            raise ValueError

        ws: list[ndarray] = []
        ps: list[ndarray] = []
        for fn, (rn, unit) in zip(component_names.keys(), units.items()):
            # The stream width
            if fn in width0:
                wn0 = width0[fn].to_value(unit)
            elif rn in width0:
                wn0 = width0[rn].to_value(unit)
            else:
                msg = f"{fn}/{rn} are not in the stream data, setting the width to 0."
                wn0 = 0

            ws.append(np.array([[wn0**2, 0], [0, 0]]))

            # The R contribution to the error
            # there are 2 options, the frame or the rep component name.
            if (fne := f"{fn}_err") in arm.data.columns:
                rn0 = arm.data[fne][:3].mean().to_value(unit)
            elif (rne := f"{rn}_err") in arm.data.columns:
                rn0 = arm.data[rne][:3].mean().to_value(unit)
            else:
                msg = f"{fne}/{rne} are not in the stream data, setting the error to the width."
                warnings.warn(msg)
                rn0 = 0

            # combine data error with stream width
            pn = rn0**2 + wn0**2

            # covariance block
            # p = np.array([[pn, 0], [0, 10 * pn]])
            p = np.array([[pn, 0], [0, pn]])
            ps.append(p)

        P0 = block_diag(*ps)  # Covariance matrix
        # Q0 = block_diag(*ws)  # Process Noise Model
        Q0 = np.zeros_like(P0)

        return cls(
            x0=x0,
            P0=P0,
            Q0=Q0,
            onsky=onsky,
            kinematics=kinematics,
            frame=frame,
            frame_representation_type=rep_type,
            frame_differential_type=dif_type,
            units=units,
            options=options,
        )

    # ---------------------------------------------------------------

    @property
    def ndims(self) -> int:
        """Total number of dimensions of the Kalman Filter.

        See Also
        --------
        ndims_rep
        ndims_dif
        """
        return len(self.x0) // 2

    @property
    def ndims_rep(self) -> int:
        """Total number of positional dimensions of the Kalman Filter.

        This should equal
        `trackstream.fit.kalman.FirstOrderNewtonianKalmanFilter.ndims`
        if there are `trackstream.fit.kalman.FirstOrderNewtonianKalmanFilter.kinematics`
        is `False`.

        See Also
        --------
        ndims
        ndims_dif
        """
        rac = getattr(self.frame_representation_type, "attr_classes", ())
        nrepdims: int = len(rac)
        return nrepdims

    @property
    def ndims_dif(self) -> int:
        if not self.kinematics:
            return 0

        dac = getattr(self.frame_differential_type, "attr_classes", ())
        nrepdims: int = len(dac)
        return nrepdims

    @property
    def _frame_dif_attrs(self) -> ValuesView[str]:
        attrs = super()._frame_dif_attrs if self.kinematics else {}.values()
        return attrs

    @property
    def frame_component_names(self) -> dict[str, str]:
        component_names = self.frame.get_representation_component_names()
        if self.kinematics:
            component_names.update(self.frame.get_representation_component_names("s"))

        return component_names

    # ---------------------------------------------------------------
    # Initial Conditions

    def state_transition_model(self, dt: Quantity) -> ndarray:
        """Make Transition Matrix.

        Parameters
        ----------
        dt : scalar or (N,) structured Quantity
            Time step or array thereof. Must be a structured Quantity with
            fields ``positions`` and ``kinematics`` (if
            `~trackstream.fit.kalman.FirstOrderNewtonianKalmanFilter.kinematics`
            is `True`).

        Returns
        -------
        F : ndarray
            Block diagonal transition matrix. The shape will be (1, ndims,
            ndims) or (N, ndims, ndims), depending if ``dt`` was scalar or of
            length ``N``.
        """
        # # make single block of F matrix
        # # [[position to position, position from velocity]
        # #  [velocity from position, velocity to velocity]]
        # f = np.array([[1.0, dt], [0, 1.0]])
        # # F block-diagonal array
        # F: ndarray = block_diag(*([f] * self.ndims))
        # return F
        F = np.zeros((np.size(dt), 2 * self.ndims, 2 * self.ndims))
        idx = np.arange(2 * self.ndims)[::2]

        F[:, idx, idx] = 1  # positions
        F[:, idx + 1, idx + 1] = 1  # kf `velocities'

        # time steps, iterating over x and v (skipped if 0).
        i = 0  # global counter over both for-loops
        for n, d in zip(("positions", "kinematics"), (self.ndims_rep, self.ndims_dif)):
            for _ in range(d):
                F[:, 2 * i, 2 * i + 1] = cast(Quantity, dt[n]).to_value(self.units[i])
                i += 1

        return F

    # # TODO check against Q_discrete_white_noise
    # # from filterpy.common import Q_discrete_white_noise
    def process_noise_model(self, dt: Quantity, var: float = 1.0) -> ndarray:
        """Process noise.

        Returns
        -------
        (D, D) ndarray
            The ``Q`` term of a Kalman filter.
        """
        Q = np.zeros((np.size(dt), 2 * self.ndims, 2 * self.ndims))

        # Fill in block-diagonal
        # single-component of Q matrix
        #     [[0.25 * dt**4, 0.5 * dt**3],  # 1,1 is position
        #      [0.5 * dt**3, dt**2]]  # 2,2 is kf-`velocity'
        i = 0  # global counter over both for-loops
        for n, d in zip(("positions", "kinematics"), (self.ndims_rep, self.ndims_dif)):
            for _ in range(d):
                _dt = cast(Quantity, dt[n]).to_value(self.units[i])

                Q[:, 2 * i, 2 * i] = 0.25 * _dt**4
                Q[:, 2 * i, 2 * i + 1] = 0.5 * _dt**3
                Q[:, 2 * i + 1, 2 * i] = 0.5 * _dt**3
                Q[:, 2 * i + 1, 2 * i + 1] = _dt**2

                i += 1

        return var * Q

    # ---------------------------------------------------------------
    # Coordinates <-> internals

    def _get_value_from_coord(self, data: CoordinateType, /) -> ndarray:
        """Coordinate / Representation to unstructured array.

        Parameters
        ----------
        crd : (N,) SkyCoord or BaseCoordinateFrame or BaseRepresentation, positional-only
            Coordinates to take an `numpy.ndarray`.

        Returns
        -------
        (N, D) ndarray
        """
        # Representation
        crd = data.transform_to(self.frame)
        rep = crd.represent_as(
            self.frame_representation_type,
            s=self.frame_differential_type if self.kinematics else None,
            in_frame_units=True,
        )

        # rep -> structured array
        sv, su = _rep_to_array(rep)
        # unstructure and get conversion factor
        uv = structured_to_unstructured(sv)
        uu = structured_to_unstructured(su.to(self.units))

        return uv * uu

    def _array_to_structured_quantity(self, v: ndarray, /) -> Quantity:
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
        dt = np.dtype([(n, float) for n in self.units.field_names])
        # Make structured quantity
        q: Quantity = unstructured_to_structured(v, dt) << self.units
        return q

    def _array_to_coord(self, v: ndarray, /) -> SkyCoord:
        """array to SkyCoord.

        Parameters
        ----------
        v : ndarray
            The data to convert to a SkyCoord

        Returns
        -------
        SkyCoord
        """
        q: Quantity = self._array_to_structured_quantity(v)

        # Differentials
        if self.kinematics:
            dif = self.frame_differential_type(**{n: q[n] for n in self._frame_dif_attrs}, copy=False)
        else:
            dif = None

        # Representation, including Differentials
        rcs = {n: q[n] for n in self._frame_rep_attrs}
        rep = self.frame_representation_type(**rcs, differentials=dif, copy=False)

        # Frame
        crd: BaseCoordinateFrame = self.frame.realize_frame(
            rep,
            representation_type=self.frame_representation_type,
            differential_type=self.frame_differential_type,
            copy=False,
        )

        return SkyCoord(crd, copy=False)

    def _wrap_residual(self, residual: ndarray) -> ndarray:
        #  first coordinate is always the longitude
        deltalon = residual[0] * self._angle_convert[0]
        pa = arctan2(sin(deltalon), 0)  # position angle
        residual[0] = sign(pa) * arccos(cos(deltalon)) / self._angle_convert[0]

        # TODO! similar for |Latitude|

        return residual

    def _wrap_posterior(self, x: ndarray) -> ndarray:
        # first coordinate is always the longitude
        # keeps in (-180, 180) deg
        wlon, wlat = self._wrap_at
        x[0] += 2 * wlon if (x[0] < -wlon) else 0
        x[0] -= 2 * wlon if (x[0] >= wlon) else 0

        # # similar unwrapping of the |Latitude|
        # x[1] += wlat if (x[1] < -wlat) else 0
        # x[1] -= wlat if (x[1] >= wlat) else 0

        return x

    #######################################################
    # Math (2 phase + smoothing)

    def _math_predict_and_update(
        self, x: ndarray, P: ndarray, F: ndarray, Q: ndarray, z: ndarray, R: ndarray
    ) -> tuple[ndarray, ndarray]:
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

        S = dot(dot(self.H, P), self.H.T) + R  # system uncertainty in measurement space
        K = dot(dot(P, self.H.T), inv(S))  # Kalman gain

        predict = dot(self.H, x)  # prediction in measurement space
        residual = z - predict  # measurement and prediction residual
        if self.onsky:  # unwrap to keep x close
            residual = self._wrap_residual(residual)

        x = x + dot(K, residual)  # predict new x using Kalman gain

        if self.onsky:  # need to correct for phase-wrap
            x = self._wrap_posterior(x)

        KH = dot(K, self.H)
        ImKH = self._I - KH
        # stable representation using Joseph equation (from Filterpy)
        # P = (1 - KH)P(1 - KH)' + KRK'
        P = dot(dot(ImKH, P), ImKH.T) + dot(dot(K, R), K.T)

        return x, P

    def _rts_smoother(self, Xs: ndarray, Ps: ndarray, Fs: ndarray, Qs: ndarray) -> tuple[ndarray, ndarray]:
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
            # TODO! limite Kalman gain
            K[k] = dot(dot(P[k], F.T), inv(Pp[k]))

            # update position and covariance
            x_prior = dot(F, x[k])
            residual = x[k + 1] - x_prior
            if self.onsky:  # unwrap to keep x close
                residual = self._wrap_residual(residual)

            x[k] += dot(K[k], residual)
            P[k] += dot(dot(K[k], P[k + 1] - Pp[k]), K[k].T)

            if self.onsky:  # need to correct for phase-wrap
                x[k][:] = self._wrap_posterior(x[k])

        return x, P

    #######################################################
    # Fit

    def fit(
        self,
        data: SkyCoord,
        /,
        errors: Quantity,
        widths: Quantity,
        timesteps: Quantity,
        *,
        name: str | None = None,
    ) -> Path:
        """Run Kalman Filter with updates on each step.

        Parameters
        ----------
        data : (N,) SkyCoord, position-only
            The data to fit with the Kalman filter.
        errors : (N,) Quantity
            A structured Quantity.
            Field names should be the representation names.
        width : (N,) Quantity
            A structured Quantity.
            Field names should be the representation names.

        timesteps: (N,) ndarray
            Must be start and end-point inclusive.

        Returns
        -------
        path : `~trackstream.fit.path.Path`
        """
        # ------ setup ------

        # Get the time deltas from the time steps.
        # Checking the time steps are compatible with the data.
        if len(timesteps) != len(data):
            raise ValueError(f"len(timesteps)={len(timesteps)} is not {len(data)}")
        elif np.any(apply_along_fields(is_negative, timesteps.value)):
            raise ValueError("timesteps must be >= 0")

        # Measurements (in correct units).
        Zs = self._get_value_from_coord(data)
        num_z = len(Zs)

        # Error matrix
        # TODO! it would be great to be able to transform errors
        # as well. For now, the errors must be in kalman filter's rep/diff type and units
        idx = np.arange(self.ndims)  # diagonal indices
        errs = structured_to_unstructured(errors.to_value(self.units))  # unstructure
        Rs = np.zeros((len(errors), self.ndims, self.ndims))
        Rs[:, idx, idx] = errs**2  # assign to diagonal

        # Widths
        ws = structured_to_unstructured(widths.to_value(self.units))  # unstructure
        Ws = np.zeros((len(errors), self.ndims))
        Ws[:] = ws**2

        # ------ IC (i-1) ------

        x, P = self.x0, self.P0  # KF

        # initialize arrays
        Xs = empty((num_z, *np.shape(x)))
        Ps = empty((num_z, *np.shape(P)))

        # Make the transition model and process noise model
        Fs = self.state_transition_model(timesteps)
        Qs = self.process_noise_model(timesteps, var=1)

        # ------ run ------
        # iterate predict & update steps
        z: ndarray
        dt: Quantity
        for i, (z, R, F, Q, dt) in enumerate(zip(Zs, Rs, Fs, Qs, timesteps)):
            # F_(i-1, i)
            R[idx, idx] += Ws[i]  # add stream width to uncertainty
            # # TODO! this is at the previous step! need to
            # # do it during predict / update, not before
            # R[idx, idx] += widths(x[::2][: self.ndims_rep], x[::2][self.ndims_rep :])  # or z?

            # predict & update
            x, P = self._math_predict_and_update(x=x, P=P, F=F, Q=Q, z=z, R=R)

            # append results
            Xs[i], Ps[i] = x, P

        # save
        result = kalman_output(timesteps, Xs, Ps)

        # smoothed
        xs, Ps = self._rts_smoother(Xs, Ps, Fs, Qs)
        smooth = kalman_output(timesteps, xs, Ps)

        # ------ make path ------

        # Measured Xs
        # mXs = dot(self.H, smooth.x.T).T  # (N, D)
        # needs to be C-continuous
        mc = self._array_to_coord(np.array(smooth.x[:, ::2], copy=True))

        # Covariance matrix.
        # We only care about the real coordinates, not the KF `velocity'.
        # The block elements are uncorrelated.  # TODO! more general
        cov = smooth.P[:, ::2, ::2]
        var = np.diagonal(cov, axis1=1, axis2=2)
        stream_width = self._array_to_structured_quantity(np.sqrt(var))

        # Affine  # TODO! is this the Affine wanted?
        ci = cast(SkyCoord, mc[:-1])
        cj = cast(SkyCoord, mc[1:])
        sp2p = ci.separation(cj)  # point-2-point sep
        minafn = Quantity(min(1e-10, 1e-10 * sp2p.value[0]), sp2p.unit)
        affine = np.concatenate((np.atleast_1d(np.array(minafn, subok=True)), sp2p.cumsum()))

        path = Path.from_skycoord(
            mc,
            width=stream_width,
            amplitude=None,  # TODO!
            name=name,
            affine=Quantity(affine, copy=False),
            meta=dict(result=result, smooth=smooth),
        )

        return path
