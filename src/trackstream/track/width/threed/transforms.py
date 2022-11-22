"""Transforms for one-dimensional widths."""

from __future__ import annotations

# STDLIB
from typing import TYPE_CHECKING, cast

# THIRD PARTY
import astropy.units as u
import numpy as np
from astropy.coordinates import BaseRepresentation, SphericalRepresentation

# LOCAL
from trackstream.track.width.threed.core import (
    Cartesian3DWidth,
    SphericalWidth,
    UnitSphericalWidth,
)
from trackstream.track.width.transforms import register_transformation

if TYPE_CHECKING:
    # THIRD PARTY
    from astropy.units import Quantity

__all__: list[str] = []


##############################################################################
# CODE
##############################################################################


@register_transformation(Cartesian3DWidth, UnitSphericalWidth)
def cartesian_to_unitspherical(cw: Cartesian3DWidth, point: BaseRepresentation) -> UnitSphericalWidth:
    """Convert a Cartesian3DWidth to a UnitSphericalWidth.

    Parameters
    ----------
    cw : Cartesian3DWidth
        The Cartesian3DWidth to convert.
    point : BaseRepresentation
        The point at which to convert the width.

    Returns
    -------
    UnitSphericalWidth
        The converted width.
    """
    # FIXME! actual projection. This is a bad approx.
    w = cast("Quantity", np.sqrt(cw.x**2 + cw.y**2 + cw.z**2))
    spnt = cast("SphericalRepresentation", point.represent_as(SphericalRepresentation))
    distance = spnt.distance.to_value(w.unit)
    sw = np.abs(np.arctan2(w.value, distance)) << u.rad

    return UnitSphericalWidth(sw, sw)  # (lat, lon)


@register_transformation(Cartesian3DWidth, SphericalWidth)
def cartesian_to_spherical(cw: Cartesian3DWidth, point: BaseRepresentation) -> SphericalWidth:
    """Convert a Cartesian3DWidth to a SphericalWidth.

    Parameters
    ----------
    cw : Cartesian3DWidth
        The width to convert.
    point : BaseRepresentation
        The point at which to convert the width.

    Returns
    -------
    SphericalWidth
        The converted width.
    """
    usw = cartesian_to_unitspherical(cw, point)
    distance = cast("Quantity", np.sqrt(cw.x**2 + cw.y**2 + cw.z**2))  # FIXME! This is a bad approx.

    return SphericalWidth(lat=usw.lat, lon=usw.lon, distance=distance)
