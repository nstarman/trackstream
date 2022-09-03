"""Kalman Filter code."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from dataclasses import dataclass

# THIRD PARTY
import astropy.coordinates as coords
import astropy.units as u

# LOCAL
from trackstream.track.fit.kalman.base import FONKFBase, KFInfo

__all__: list[str] = []


##############################################################################
# CODE
##############################################################################


@dataclass(frozen=True)
class CartesianFONKF(FONKFBase):
    info = KFInfo(
        representation_type=coords.CartesianRepresentation,
        differential_type=coords.CartesianDifferential,
        units=u.StructuredUnit(
            ((u.kpc, u.kpc, u.kpc), (u.km / u.s, u.km / u.s, u.km / u.s)),
            names=(("x", "y", "z"), ("d_x", "d_y", "d_z")),
        ),
    )
