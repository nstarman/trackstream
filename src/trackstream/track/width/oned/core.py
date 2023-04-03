"""One-dimensionsal widths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from astropy.coordinates import RadialDifferential, RadialRepresentation
from astropy.units import Quantity  # noqa: TCH002

from trackstream.track.width.core import (
    LENGTH,
    SPEED,
    BaseWidth,
    ConfigSpaceWidth,
    KinematicSpaceWidth,
)
from trackstream.utils.descriptors.classproperty import classproperty

__all__: list[str] = []

if TYPE_CHECKING:
    from astropy.units import PhysicalType


##############################################################################
# CODE
##############################################################################


@dataclass(frozen=True)
class Cartesian1DWidth(ConfigSpaceWidth):
    """1D Cartesian width."""

    x: Quantity  # [LENGTH | ANGLE]

    @classproperty
    def corresponding_representation_type(cls) -> type[RadialRepresentation]:
        """Representation type corresponding to the width type."""
        return RadialRepresentation

    @property
    def corresponding_width_types(cls) -> dict[PhysicalType, None | type[BaseWidth]]:
        """The width types corresponding to this width type."""
        return {LENGTH: cls.__class__, SPEED: Cartesian1DiffWidth}


@dataclass(frozen=True)
class AngularWidth(ConfigSpaceWidth):
    """1D angular width in configuration space."""

    lat: Quantity  # [ANGLE]

    @classproperty
    def corresponding_representation_type(cls) -> type[RadialRepresentation]:
        """Representation type corresponding to the width type."""
        return RadialRepresentation

    @property
    def corresponding_width_types(cls) -> dict[PhysicalType, None | type[BaseWidth]]:
        """The width types corresponding to this width type."""
        return {LENGTH: cls.__class__, SPEED: AngularDiffWidth}


# ===================================================================
# Kinematics


@dataclass(frozen=True)
class Cartesian1DiffWidth(KinematicSpaceWidth):
    """1D width in velocity space."""

    d_x: Quantity  # [SPEED | ANGULAR_SPEED]

    @classproperty
    def corresponding_representation_type(cls) -> type[RadialDifferential]:
        """Representation type corresponding to the width type."""
        return RadialDifferential

    @property
    def corresponding_width_types(self) -> dict[PhysicalType, None | type[BaseWidth]]:
        """The width types corresponding to this width type."""
        return {LENGTH: Cartesian1DWidth, SPEED: self.__class__}


@dataclass(frozen=True)
class AngularDiffWidth(KinematicSpaceWidth):
    """1D angular width in velocity space."""

    d_lat: Quantity  # [ANGULAR_SPEED]

    @classproperty
    def corresponding_representation_type(cls) -> type[RadialDifferential]:
        """Representation type corresponding to the width type."""
        return RadialDifferential

    @property
    def corresponding_width_types(self) -> dict[PhysicalType, None | type[BaseWidth]]:
        """The width types corresponding to this width type."""
        return {LENGTH: AngularWidth, SPEED: self.__class__}
