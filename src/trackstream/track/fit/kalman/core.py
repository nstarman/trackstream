"""Kalman Filter code."""

from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any, cast

from astropy.coordinates import BaseCoordinateFrame  # noqa: TCH002
import astropy.units as u
import numpy as np
import numpy.lib.recfunctions as rfn

from trackstream.stream.base import FRAME_NONE_ERR
from trackstream.stream.core import StreamArm
from trackstream.track.fit.exceptions import EXCEPT_3D_NO_DISTANCES
from trackstream.track.fit.kalman.builtin import CartesianFONKF, USphereFONKF
from trackstream.track.fit.utils import _c2v, _v2c
from trackstream.track.path import Path
from trackstream.track.width.core import BASEWIDTH_REP as _PW_REP
from trackstream.track.width.core import LENGTH, SPEED
from trackstream.track.width.plural import Widths
from trackstream.utils.unit_utils import merge_units

__all__: list[str] = []


if TYPE_CHECKING:
    from astropy.coordinates import SkyCoord
    from typing_extensions import Self

    from trackstream._typing import NDFloating
    from trackstream.track.fit.kalman.base import FONKFBase, KFInfo
    from trackstream.track.fit.timesteps.plural import Times


##############################################################################
# CODE
##############################################################################


@dataclass(frozen=True)
class FirstOrderNewtonianKalmanFilter:
    """First-order Newtonian Kalman filter class.

    This is a wrapper class around
    `trackstream.track.fit.kalman.base.FONKFBase`, the low-level Kalman Filter
    implementation. The easiest way to instantiate this class is with
    ``from_format``, which should correctly make the low-level object.

    Parameters
    ----------
    kf : `trackstream.track.fit.kalman.base.FONKFBase`
        Low-level kalman filter implementation.
    frame : BaseCoordinateFrame
        The frame of the data.

    """

    kf: FONKFBase
    """Low-level kalman filter implementation."""

    frame: BaseCoordinateFrame
    """Frame of the Kalman filter."""

    # ===============================================================
    # Convenience hooks on the low-level object.

    @property
    def x0(self) -> BaseCoordinateFrame:
        """Return the initial position."""
        return _v2c(self, self.kf.x0)

    @property
    def P0(self) -> NDFloating:
        """Return the initial covariance."""
        return self.kf.P0

    @property
    def Q0(self) -> NDFloating | None:
        """Return the process noise."""
        return self.kf.Q0

    @property
    def info(self) -> KFInfo:
        """Information for interfacing with frame information."""
        return self.kf.info

    @property
    def onsky(self) -> bool:
        """If the KF works o nthe sky or in 3D."""
        return bool(isinstance(self.kf, USphereFONKF))

    @property
    def kinematics(self) -> bool:
        """If the KF works on the kinematics."""
        # two options for number of features
        kfs = (CartesianFONKF, USphereFONKF)
        nf = (6, 4)
        i = kfs.index(type(self.kf))
        return bool(self.nfeature == nf[i])

    @property
    def nfeature(self) -> int:
        """Total number of dimensions of the Kalman Filter."""
        return self.kf.nfeature

    # ===============================================================

    @singledispatchmethod
    @classmethod
    def from_format(
        cls,
        arm: object,  # noqa: ARG003
        onsky: bool | None = None,  # noqa: ARG003
        kinematics: bool | None = None,  # noqa: ARG003
        *,
        width0: None | Widths = None,  # noqa: ARG003
    ) -> Any:  # https://github.com/python/mypy/issues/11727
        """Create a Kalman filter from an object.

        Parameters
        ----------
        arm : object
            The object to create the Kalman filter from.
        onsky : bool | None, optional
            If the Kalman filter should be on the sky or in 3D.
        kinematics : bool | None, optional
            If the Kalman filter should include the kinematics.
        width0 : None | Widths, optional
            Initial widths for the Kalman filter.

        Returns
        -------
        FirstOrderNewtonianKalmanFilter | Any
            The Kalman filter or similar object, depending on the input, over which this method is dispatched.

        """
        msg = "not dispatched"
        raise NotImplementedError(msg)

    @from_format.register(StreamArm)
    @classmethod
    def _from_format_streamarm(
        cls: type[Self],
        arm: StreamArm,
        onsky: bool | None = None,
        kinematics: bool | None = None,
        *,
        width0: None | Widths = None,
    ) -> Self:
        """Make Kalman Filter from a stream.

        Parameters
        ----------
        arm : `trackstream.stream.StreamArm`
            The stream (arm) from which to build the Kalman filter

        onsky : bool | None, optional
            If the Kalman filter should be on the sky or in 3D.
        kinematics : bool | None, optional
            If the Kalman filter should include the kinematics.

        width0 : None | Widths, optional keyword-only
            Initial widths for the Kalman filter.

        Returns
        -------
        `trackstream.fit.kalman.FONKFBase`

        """
        # flags
        if onsky is None:
            onsky = not arm.has_distances
        elif onsky is False and not arm.has_distances:
            raise EXCEPT_3D_NO_DISTANCES

        if arm.frame is None:
            raise FRAME_NONE_ERR

        frame = arm.frame
        kf_cls = USphereFONKF if onsky else CartesianFONKF

        # Width
        flat_units = merge_units(kf_cls.info.units)
        if isinstance(width0, Widths):
            pass
        elif width0 is None:
            w0q = u.Quantity(np.zeros((), dtype=[(n, float) for n in flat_units.field_names]), flat_units)
            wclss = (_PW_REP[kf_cls.info.representation_type], _PW_REP[kf_cls.info.differential_type])
            width0 = Widths(
                {
                    wcls.dimensions._physical_type_list[0]: wcls.from_format(w0q)  # noqa: SLF001
                    for wcls in wclss
                    if wcls.dimensions is not None
                },
            )
        else:
            raise ValueError

        widths0 = rfn.merge_arrays(width0.to_format(u.Quantity), flatten=True)

        # lower-level implementation
        kf = kf_cls.from_format(arm, kinematics=kinematics, width0=widths0)

        return cls(kf, frame=frame)

    # ===============================================================

    def fit(  # noqa: PLR0913
        self,
        data: SkyCoord,
        /,
        errors: u.Quantity,
        widths: Widths,
        timesteps: Times,
        *,
        name: str | None = None,
    ) -> Path:
        """Run Kalman Filter with updates on each step.

        Parameters
        ----------
        self : FirstOrderNewtonianKalmanFilter
            The Kalman filter to run.
        data : (N,) SkyCoord, positional-only
            The data to fit with the Kalman filter.
        errors : (N,) Quantity
            A structured Quantity.
            Field names should be the representation names.
        widths : (N,) Widths
            A structured Quantity.
            Field names should be the representation names.
        timesteps: Times
            Must be start and end-point inclusive.

        name : str | None, optional keyword-only
            The name of the path.

        Returns
        -------
        `~trackstream.fit.path.Path`

        """
        # Measurements (in correct units).
        zs = _c2v(self, data)[:, : self.nfeature]
        # Errors as (N, D) array
        errs = rfn.structured_to_unstructured(errors.to_value(merge_units(self.info.units)))[:, : self.nfeature]

        # Widths: start with (N, 2) -> (N, D)
        nd = self.nfeature // 2 if self.kinematics else self.nfeature
        _ws = (np.tile(rfn.structured_to_unstructured(np.array(widths["length"])), (1, nd)),)
        if self.kinematics:
            _ws += (np.tile(rfn.structured_to_unstructured(np.array(widths["speed"])), (1, nd)),)
        ws = np.c_[_ws]

        # timesteps
        tu = (
            u.StructuredUnit((self.kf.info.units[0][0], self.kf.info.units[1][0]))
            if self.kinematics
            else self.kf.info.units[0][0]
        )
        dts = rfn.structured_to_unstructured(timesteps.to_format(u.Quantity).to_value(tu))

        result, smooth = self.kf.fit(zs, errors=errs, widths=ws, timesteps=dts)

        # ------ make path ------

        # Measured Xs
        # needs to be C-continuous
        mc = _v2c(self, np.array(smooth.x[:, ::2], copy=True))

        # Covariance matrix.
        # We only care about the real coordinates, not the KF `velocity'.
        # The block elements are uncorrelated.
        cov = smooth.P[:, ::2, ::2]
        var = np.diagonal(cov, axis1=1, axis2=2)
        std = np.sqrt(var)

        if self.kinematics:
            sws = u.Quantity(rfn.unstructured_to_structured(std, dtype=self.info.dtype), unit=self.info.units)
            dws = {
                LENGTH: _PW_REP[self.info.representation_type].from_format(sws["length"]),
                SPEED: _PW_REP[self.info.differential_type].from_format(sws["speed"]),
            }

        else:
            sws = u.Quantity(rfn.unstructured_to_structured(std, dtype=self.info.dtype[0]), unit=self.info.units[0])
            dws = {LENGTH: _PW_REP[self.info.representation_type].from_format(sws)}

        stream_width = Widths(dws)

        # Affine
        ci = cast("BaseCoordinateFrame", mc[:-1])
        cj = mc[1:]
        sp2p = ci.separation(cj)  # point-2-point sep
        minafn = u.Quantity(min(1e-10, 1e-10 * sp2p.value[0]), sp2p.unit)
        affine = u.Quantity(np.concatenate((np.atleast_1d(np.array(minafn, subok=True)), sp2p.cumsum())), copy=False)

        return Path.from_format(
            mc,
            width=stream_width,
            amplitude=None,  # TODO: implement this
            name=name,
            affine=affine,
            metadata={"result": result, "smooth": smooth},
        )
