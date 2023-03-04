"""Three-dimensional widths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import astropy.coordinates as coords
from astropy.units import Quantity  # noqa: TCH002

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

if TYPE_CHECKING:
    from astropy.units import PhysicalType

##############################################################################
# CODE
##############################################################################


@dataclass(frozen=True)
class Cartesian3DWidth(Cartesian2DWidth):
    """3D Cartesian width in configuration space."""

    z: Quantity  # [LENGTH]

    @classproperty
    def corresponding_representation_type(cls) -> type[coords.CartesianRepresentation]:
        """Representation type corresponding to the width type."""
        return coords.CartesianRepresentation

    @property
    def corresponding_width_types(self) -> dict[PhysicalType, None | type[BaseWidth]]:
        """The width types corresponding to this width type."""
        return {LENGTH: self.__class__, SPEED: Cartesian2DiffWidth}


@dataclass(frozen=True)
class UnitSphericalWidth(AngularWidth):
    """3D-embedded 2D on-sky spherical representation in configuration space."""

    lon: Quantity  # [ANGLE]
    lat: Quantity  # [ANGLE]

    @classproperty
    def corresponding_representation_type(cls) -> type[coords.UnitSphericalRepresentation]:
        """Representation type corresponding to the width type."""
        return coords.UnitSphericalRepresentation

    @property
    def corresponding_width_types(self) -> dict[PhysicalType, None | type[BaseWidth]]:
        """The width types corresponding to this width type."""
        return {LENGTH: self.__class__, SPEED: UnitSphericalDiffWidth}


@dataclass(frozen=True)
class SphericalWidth(UnitSphericalWidth, PolarWidth):
    """3D spherical representation in configuration space."""

    lon: Quantity  # [ANGLE]
    lat: Quantity  # [ANGLE]
    distance: Quantity  # [LENGTH]

    @classproperty
    def corresponding_representation_type(cls) -> type[coords.SphericalRepresentation]:
        """Representation type corresponding to the width type."""
        return coords.SphericalRepresentation

    @property
    def corresponding_width_types(self) -> dict[PhysicalType, None | type[BaseWidth]]:
        """The width types corresponding to this width type."""
        return {LENGTH: self.__class__, SPEED: SphericalDiffWidth}


# ===================================================================
# Kinematic Space


@dataclass(frozen=True)
class Cartesian3DiffWidth(Cartesian2DiffWidth):
    """3D Cartesian width in velocity space."""

    d_z: Quantity  # [SPEED]

    @property
    def corresponding_width_types(self) -> dict[PhysicalType, None | type[BaseWidth]]:
        """The width types corresponding to this width type."""
        return {LENGTH: Cartesian3DWidth, SPEED: self.__class__}

    @classproperty
    def corresponding_representation_type(cls) -> type[coords.CartesianDifferential]:
        """Representation type corresponding to the width type."""
        return coords.CartesianDifferential


@dataclass(frozen=True)
class UnitSphericalDiffWidth(AngularDiffWidth):
    """3D-embedded 2D on-sky spherical width in velocity space."""

    d_lon: Quantity  # [ANGULAR_SPEED]
    d_lat: Quantity  # [ANGULAR_SPEED]

    @classproperty
    def corresponding_representation_type(cls) -> type[coords.UnitSphericalDifferential]:
        """Representation type corresponding to the width type."""
        return coords.UnitSphericalDifferential

    @property
    def corresponding_width_types(self) -> dict[PhysicalType, None | type[BaseWidth]]:
        """The width types corresponding to this width type."""
        return {LENGTH: UnitSphericalWidth, SPEED: self.__class__}


@dataclass(frozen=True)
class SphericalDiffWidth(UnitSphericalWidth, PolarDiffWidth):
    """3D spherical width in velocity space."""

    d_lon: Quantity  # [ANGULAR_SPEED]
    d_lat: Quantity  # [ANGULAR_SPEED]
    d_distance: Quantity  # [SPEED]

    @classproperty
    def corresponding_representation_type(cls) -> type[coords.SphericalDifferential]:
        """Representation type corresponding to the width type."""
        return coords.SphericalDifferential

    @property
    def corresponding_width_types(self) -> dict[PhysicalType, None | type[BaseWidth]]:
        """The width types corresponding to this width type."""
        return {LENGTH: SphericalWidth, SPEED: self.__class__}
