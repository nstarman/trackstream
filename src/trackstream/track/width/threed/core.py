from __future__ import annotations

# STDLIB
from dataclasses import dataclass

# THIRD PARTY
import astropy.coordinates as coords
import astropy.units as u

# LOCAL
from trackstream.track.width.core import LENGTH, SPEED, BaseWidth
from trackstream.track.width.oned.core import AngularDiffWidth, AngularWidth
from trackstream.track.width.twod.core import (
    Cartesian2DiffWidth,
    Cartesian2DWidth,
    PolarDiffWidth,
    PolarWidth,
)
from trackstream.utils.descriptors.classproperty import classproperty

__all__: list[str] = []

##############################################################################
# CODE
##############################################################################


@dataclass(frozen=True)
class Cartesian3DWidth(Cartesian2DWidth):
    """3D Cartesian width in configuration space."""

    z: u.Quantity  # [LENGTH]

    @classproperty
    def corresponding_representation_type(cls) -> type[coords.CartesianRepresentation]:
        return coords.CartesianRepresentation

    @property
    def corresponding_width_types(self) -> dict[u.PhysicalType, None | type[BaseWidth]]:
        return {LENGTH: self.__class__, SPEED: Cartesian2DiffWidth}


@dataclass(frozen=True)
class UnitSphericalWidth(AngularWidth):
    """3D-embedded 2D on-sky spherical representation in configuration space."""

    lon: u.Quantity  # [ANGLE]
    lat: u.Quantity  # [ANGLE]

    @classproperty
    def corresponding_representation_type(cls) -> type[coords.UnitSphericalRepresentation]:
        return coords.UnitSphericalRepresentation

    @property
    def corresponding_width_types(self) -> dict[u.PhysicalType, None | type[BaseWidth]]:
        return {LENGTH: self.__class__, SPEED: UnitSphericalDiffWidth}


@dataclass(frozen=True)
class SphericalWidth(UnitSphericalWidth, PolarWidth):
    """3D spherical representation in configuration space."""

    lon: u.Quantity  # [ANGLE]
    lat: u.Quantity  # [ANGLE]
    distance: u.Quantity  # [LENGTH]

    @classproperty
    def corresponding_representation_type(cls) -> type[coords.SphericalRepresentation]:
        return coords.SphericalRepresentation

    @property
    def corresponding_width_types(self) -> dict[u.PhysicalType, None | type[BaseWidth]]:
        return {LENGTH: self.__class__, SPEED: SphericalDiffWidth}


# ===================================================================
# Kinematic Space


@dataclass(frozen=True)
class Cartesian3DiffWidth(Cartesian2DiffWidth):
    """3D Cartesian width in velocity space."""

    d_z: u.Quantity  # [SPEED]

    @property
    def corresponding_width_types(self) -> dict[u.PhysicalType, None | type[BaseWidth]]:
        return {LENGTH: Cartesian3DWidth, SPEED: self.__class__}

    @classproperty
    def corresponding_representation_type(cls) -> type[coords.CartesianDifferential]:
        return coords.CartesianDifferential


@dataclass(frozen=True)
class UnitSphericalDiffWidth(AngularDiffWidth):
    """3D-embedded 2D on-sky spherical width in velocity space."""

    d_lon: u.Quantity  # [ANGULAR_SPEED]
    d_lat: u.Quantity  # [ANGULAR_SPEED]

    @classproperty
    def corresponding_representation_type(cls) -> type[coords.UnitSphericalDifferential]:
        return coords.UnitSphericalDifferential

    @property
    def corresponding_width_types(self) -> dict[u.PhysicalType, None | type[BaseWidth]]:
        return {LENGTH: UnitSphericalWidth, SPEED: self.__class__}


@dataclass(frozen=True)
class SphericalDiffWidth(UnitSphericalWidth, PolarDiffWidth):
    """3D spherical width in velocity space."""

    d_lon: u.Quantity  # [ANGULAR_SPEED]
    d_lat: u.Quantity  # [ANGULAR_SPEED]
    d_distance: u.Quantity  # [SPEED]

    @classproperty
    def corresponding_representation_type(cls) -> type[coords.SphericalDifferential]:
        return coords.SphericalDifferential

    @property
    def corresponding_width_types(self) -> dict[u.PhysicalType, None | type[BaseWidth]]:
        return {LENGTH: SphericalWidth, SPEED: self.__class__}
