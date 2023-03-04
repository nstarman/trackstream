"""Kalman Filter code."""


from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import TYPE_CHECKING

import astropy.coordinates as coords
import astropy.units as u
from numpy import arccos, arctan2, array, cos, sign, sin

from trackstream.track.fit.kalman.base import FONKFBase, KFInfo

__all__: list[str] = []

if TYPE_CHECKING:
    from trackstream._typing import NDFloating


##############################################################################
# CODE
##############################################################################


@dataclass(frozen=True)
class CartesianFONKF(FONKFBase):
    """Cartesian First-Order Newtonian Kalman Filter."""

    info = KFInfo(
        representation_type=coords.CartesianRepresentation,
        differential_type=coords.CartesianDifferential,
        units=u.StructuredUnit(
            ((u.kpc, u.kpc, u.kpc), (u.km / u.s, u.km / u.s, u.km / u.s)),
            names=(("x", "y", "z"), ("d_x", "d_y", "d_z")),
        ),
    )


@dataclass(frozen=True)
class USphereFONKF(FONKFBase):
    """Unit Sphere First-Order Newtonian Kalman Filter."""

    info = KFInfo(
        representation_type=coords.UnitSphericalRepresentation,
        differential_type=coords.UnitSphericalDifferential,
        units=u.StructuredUnit(
            ((u.rad, u.rad), (u.mas / u.yr, u.mas / u.yr)),
            names=(("lon", "lat"), ("d_lon", "d_lat")),
        ),
    )

    def __post_init__(self) -> None:
        super().__post_init__()

        self._wrap_at: NDFloating
        object.__setattr__(self, "_wrap_at", array([pi, pi / 2]))  # lon, lat [rad]

    # ===============================================================

    def _wrap_residual(self, residual: NDFloating) -> NDFloating:
        deltalon = residual[0]  # first coordinate is always the longitude
        pa = arctan2(sin(deltalon), 0)  # position angle
        residual[0] = sign(pa) * arccos(cos(deltalon))

        # TODO! similar for |Latitude|

        return residual

    def _wrap_posterior(self, x: NDFloating) -> NDFloating:
        # first coordinate is always the longitude
        # keeps in (-180, 180) deg
        wlon, wlat = self._wrap_at
        x[0] += 2 * wlon if (x[0] < -wlon) else 0
        x[0] -= 2 * wlon if (x[0] >= wlon) else 0

        # # similar unwrapping of the |Latitude|

        return x
