"""Kalman Filter code."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from typing import TYPE_CHECKING, cast

# THIRD PARTY
import astropy.coordinates as coords
import astropy.units as u
import numpy as np
from astropy.units import Quantity

# LOCAL
from trackstream.track.fit.timesteps.plural import LENGTH, SPEED, Times

if TYPE_CHECKING:
    # LOCAL
    from trackstream.track.fit.kalman.core import FirstOrderNewtonianKalmanFilter

__all__: list[str] = []


##############################################################################
# CODE
##############################################################################


def _make_timesteps(
    ds: Quantity,
    /,
    *,
    dt0: Quantity,
    dtmin: Quantity,
    dtmax: Quantity,
    width: int = 6,
) -> Quantity:
    """Make ``timesteps`` from a Quantity array.

    Parameters
    ----------
    ds : Quantity
        Steps.
    dt0 : Quantity
        Initial step.
    dtmin : Quantity
        Minimum step size.
    dtmax : Quantity or None, optional
        Maximum step size, by default None
    width : int, optional
        Kernel width, by default 6

    Returns
    -------
    Quantity
    """
    # Rolling window
    Ds = np.convolve(ds, np.ones((width,)) / width, mode="same")
    # Set minimum
    Ds[Ds < dtmin] = dtmin
    # Set maximum
    Ds[Ds > dtmax] = dtmax
    Ds = cast(Quantity, Ds)

    # munge the starts
    dts = Quantity(np.empty(len(Ds) + 1), unit=ds.unit)
    dts[1:] = Ds
    dts[0] = dt0

    return dts


def make_timesteps(
    data: coords.SkyCoord,
    /,
    kf: FirstOrderNewtonianKalmanFilter,
    *,
    dt0: Times,
    dtmin: Times | None = None,
    dtmax: Times | None = None,
    width: int = 6,
) -> Times:
    """Make distance arrays.

    Parameters
    ----------
    data : (N,) SkyCoord, position-only
        Must be ordered.

    dt0 : Quantity['length'] or Quantity['angle'], keyword-only
        Starting timestep.
    dtmin : Quantity['length'] or Quantity['angle'], keyword-only
        Minimum distance, post-convolution.

    width : int,  optional keyword-only
        Number of indices for convolution window. Default is 6.

    Returns
    -------
    timesteps : (N+1,) ndarray
        Smoothed distances, starting with 0.
    """
    # Get units. This assumes a representation has only one unit -- like
    # Cartesian or UnitSpherical.
    qu, pu = kf.info.units[0][0], kf.info.units[1][0]
    dt0["length"] <<= qu

    dtmin = Times({LENGTH: 0 * qu, SPEED: 0 * pu}) if dtmin is None else dtmin
    dtmin["length"] <<= qu

    dtmax = Times({LENGTH: np.inf * qu, SPEED: np.inf * pu}) if dtmax is None else dtmax
    dtmax["length"] <<= qu

    if kf.kinematics:
        if "speed" not in dtmin:
            dtmin["speed"] = 0 * pu

        dt0["speed"] <<= pu
        dtmin["speed"] <<= pu
        dtmax["speed"] <<= pu

    # Start with null.
    dts = Times({})
    #     {LENGTH: u.Quantity(np.nan, u.dimensionless_unscaled), SPEED: u.Quantity(np.nan, u.dimensionless_unscaled)}
    # )

    # Positions
    # point-to-point distance
    ds = getattr(data[1:], "separation" if kf.onsky else "separation_3d")(data[:-1]).to(qu)
    # now make timesteps
    dts["length"] = _make_timesteps(ds, dt0=dt0["length"], dtmin=dtmin["length"], dtmax=dtmax["length"], width=width)

    if not kf.kinematics:
        # Make array the same length to avoid padding issues.
        # dts["speed"] = (dts["length"].value * np.nan) << u.dimensionless_unscaled
        pass
    else:
        r0: coords.BaseRepresentation
        r1: coords.BaseRepresentation

        r0 = data.data[:-1]  # type: ignore
        r1 = data.data[1:]  # type: ignore

        r0cart = r0.without_differentials().to_cartesian()
        r1cart = r1.without_differentials().to_cartesian()
        ravg = ((r1cart + r0cart) / 2).represent_as(data.data.__class__)

        # There are 3 options: onsky vs not, where can either already be in cartesin coordinates or not
        d_diff = r1.differentials["s"] - r0.differentials["s"]
        ds: u.Quantity
        if kf.onsky:
            ds = d_diff.norm(ravg)
        elif isinstance(d_diff, (coords.CartesianRepresentation, coords.CartesianDifferential)):
            ds = d_diff.norm()
        else:
            ds = d_diff.to_cartesian(ravg).norm()

        ds = ds.to(pu, u.dimensionless_angles())

        dts["speed"] = _make_timesteps(ds, dt0=dt0["speed"], dtmin=dtmin["speed"], dtmax=dtmax["speed"], width=width)

    return dts
