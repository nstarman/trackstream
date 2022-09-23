from __future__ import annotations

# STDLIB
from dataclasses import dataclass

# THIRD PARTY
import astropy.coordinates as coords
import astropy.units as u

# LOCAL
from trackstream.track.width.core import LENGTH, SPEED, BaseWidth
from trackstream.track.width.oned.core import (
    AngularDiffWidth,
    AngularWidth,
    Cartesian1DiffWidth,
    Cartesian1DWidth,
)
from trackstream.utils.descriptors.classproperty import classproperty

__all__: list[str] = []

##############################################################################
# CODE
##############################################################################


@dataclass(frozen=True)
class Cartesian2DWidth(Cartesian1DWidth):
    """2D Cartesian width in configuration space."""

    y: u.Quantity  # [LENGTH]

    @classproperty
    def corresponding_representation_type(cls) -> type[coords.CartesianRepresentation]:
        return coords.CartesianRepresentation

    @property
    def corresponding_width_types(self) -> dict[u.PhysicalType, None | type[BaseWidth]]:
        return {LENGTH: self.__class__, SPEED: Cartesian2DiffWidth}


@dataclass(frozen=True)
class PolarWidth(AngularWidth):
    """2D polar representation in configuration space."""

    distance: u.Quantity  # [LENGTH]

    @classproperty
    def corresponding_representation_type(cls) -> None:
        return None

    @property
    def corresponding_width_types(self) -> dict[u.PhysicalType, None | type[BaseWidth]]:
        return {LENGTH: self.__class__, SPEED: PolarDiffWidth}


# ===================================================================
# Kinematic Space


@dataclass(frozen=True)
class Cartesian2DiffWidth(Cartesian1DiffWidth):
    """3D Cartesian width in velocity space."""

    d_y: u.Quantity  # [SPEED]

    @property
    def corresponding_width_types(self) -> dict[u.PhysicalType, None | type[BaseWidth]]:
        return {LENGTH: None, SPEED: self.__class__}

    @classproperty
    def corresponding_representation_type(cls) -> None:
        return None


@dataclass(frozen=True)
class PolarDiffWidth(AngularDiffWidth):
    """3D-embedded 2D on-sky spherical width in velocity space."""

    d_distance: u.Quantity  # [SPEED]

    @classproperty
    def corresponding_representation_type(cls) -> None:
        return None

    @property
    def corresponding_width_types(self) -> dict[u.PhysicalType, None | type[BaseWidth]]:
        return {LENGTH: None, SPEED: self.__class__}
