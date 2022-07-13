"""Kalman Filter code."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from typing import Literal, cast

# THIRD PARTY
import astropy.units as u
import numpy as np
from astropy.coordinates import CartesianDifferential, CartesianRepresentation, SkyCoord
from astropy.units import Quantity

# LOCAL
from trackstream.utils.misc import is_structured

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
    Ds = cast(Quantity, Ds)
    # Set minimum
    Ds[Ds < dtmin] = dtmin
    # Set maximum
    Ds[Ds > dtmax] = dtmax

    # munge the starts
    dts = Quantity(np.empty(len(Ds) + 1), Ds.unit)
    dts[1:] = Ds
    dts[0] = dt0

    # return cast(Quantity, np.cumsum(dts))
    return dts


def _make_timesteps_kinematics(
    data: SkyCoord,
    /,
    onsky: bool,
    *,
    dt0: Quantity,
    dtmin: Quantity | Literal[0],
    dtmax: Quantity | float = np.inf,
    width: int = 6,
) -> Quantity:
    dtmax = Quantity(dtmax, dt0.unit, copy=False)
    dtmin = Quantity(dtmin, dt0.unit, copy=False)

    # point-to-point distance
    # TODO! account for spherical geometry
    r0 = data.data[:-1]
    r1 = data.data[1:]

    r0cart = r0.without_differentials().to_cartesian()
    r1cart = r1.without_differentials().to_cartesian()
    ravg = ((r1cart + r0cart) / 2).represent_as(data.representation_type)

    # There are 3 options: onsky vs not, where can either already be in cartesin coordinates or not
    d_diff = r1.differentials["s"] - r0.differentials["s"]
    if onsky:
        ds = d_diff.norm(ravg)
    elif isinstance(d_diff, (CartesianRepresentation, CartesianDifferential)):
        ds = d_diff.norm()
    else:
        ds = d_diff.to_cartesian(ravg).norm()

    dts = _make_timesteps(ds, dt0=dt0, dtmin=dtmin, dtmax=dtmax, width=width)
    return dts


def make_timesteps(
    data: SkyCoord,
    /,
    onsky: bool,
    kinematics: bool,
    *,
    dt0: Quantity,
    dtmin: Quantity | Literal[0] = 0,
    dtmax: Quantity | float = np.inf,
    width: int = 6,
) -> Quantity:
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
    fields = ("positions", "kinematics") if kinematics else ("positions",)
    dtype = np.dtype([(n, float) for n in fields])

    if is_structured(dt0):
        dt0 = Quantity(dt0, dtype=dtype, copy=False)
    elif kinematics:
        raise ValueError("need a starting time step for the positions and kinematics")
    else:
        dt0 = Quantity(dt0, dtype=dtype, copy=False)

    dtmin = Quantity(dtmin, unit=dt0.unit, dtype=dtype, copy=False)
    dtmax = Quantity(dtmax, unit=dt0.unit, dtype=dtype, copy=False)

    dts = u.Quantity(np.full(len(data), np.nan), dtype=dtype, unit=dt0.unit)

    # Positions
    # point-to-point distance
    # TODO! instead use distance along the projection
    di = cast(SkyCoord, data[1:])
    ds = getattr(di, "separation" if onsky else "separation_3d")(data[:-1])
    # now make timesteps
    dts["positions"] = _make_timesteps(
        ds, dt0=dt0["positions"], dtmin=dtmin["positions"], dtmax=dtmax["positions"], width=width
    )

    if kinematics:
        dts["kinematics"] = _make_timesteps_kinematics(
            data,
            onsky=onsky,
            dt0=dt0["kinematics"],
            dtmin=dtmin["kinematics"],
            dtmax=dtmax["kinematics"],
            width=width,
        )

    return dts
