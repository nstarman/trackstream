"""One-dimensionsal widths."""

from __future__ import annotations

# STDLIB
from dataclasses import dataclass

# THIRD PARTY
import astropy.coordinates as coords
import astropy.units as u

# LOCAL
from trackstream.track.width.core import (
    LENGTH,
    SPEED,
    BaseWidth,
    ConfigSpaceWidth,
    KinematicSpaceWidth,
)
from trackstream.utils.descriptors.classproperty import classproperty

__all__: list[str] = []


##############################################################################
# CODE
##############################################################################


@dataclass(frozen=True)
class Cartesian1DWidth(ConfigSpaceWidth):
    """1D Cartesian width."""

    x: u.Quantity  # [LENGTH | ANGLE]

    @classproperty
    def corresponding_representation_type(cls) -> type[coords.RadialRepresentation]:
        """Representation type corresponding to the width type."""
        return coords.RadialRepresentation

    @property
    def corresponding_width_types(cls) -> dict[u.PhysicalType, None | type[BaseWidth]]:
        """The width types corresponding to this width type."""
        return {LENGTH: cls.__class__, SPEED: Cartesian1DiffWidth}


@dataclass(frozen=True)
class AngularWidth(ConfigSpaceWidth):
    """1D angular width in configuration space."""

    lat: u.Quantity  # [ANGLE]

    @classproperty
    def corresponding_representation_type(cls) -> type[coords.RadialRepresentation]:
        """Representation type corresponding to the width type."""
        return coords.RadialRepresentation

    @property
    def corresponding_width_types(cls) -> dict[u.PhysicalType, None | type[BaseWidth]]:
        """The width types corresponding to this width type."""
        return {LENGTH: cls.__class__, SPEED: AngularDiffWidth}


# ===================================================================
# Kinematics


@dataclass(frozen=True)
class Cartesian1DiffWidth(KinematicSpaceWidth):
    """1D width in velocity space."""

    d_x: u.Quantity  # [SPEED | ANGULAR_SPEED]

    @classproperty
    def corresponding_representation_type(cls) -> type[coords.RadialDifferential]:
        """Representation type corresponding to the width type."""
        return coords.RadialDifferential

    @property
    def corresponding_width_types(self) -> dict[u.PhysicalType, None | type[BaseWidth]]:
        """The width types corresponding to this width type."""
        return {LENGTH: Cartesian1DWidth, SPEED: self.__class__}


@dataclass(frozen=True)
class AngularDiffWidth(KinematicSpaceWidth):
    """1D angular width in velocity space."""

    d_lat: u.Quantity  # [ANGULAR_SPEED]

    @classproperty
    def corresponding_representation_type(cls) -> type[coords.RadialDifferential]:
        """Representation type corresponding to the width type."""
        return coords.RadialDifferential

    @property
    def corresponding_width_types(self) -> dict[u.PhysicalType, None | type[BaseWidth]]:
        """The width types corresponding to this width type."""
        return {LENGTH: AngularWidth, SPEED: self.__class__}
