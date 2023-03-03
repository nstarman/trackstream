"""Two-dimensional widths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import astropy.coordinates as coords
from astropy.units import Quantity  # noqa: TCH002

from trackstream.track.width.core import LENGTH, SPEED, BaseWidth
from trackstream.track.width.oned.core import (
    AngularDiffWidth,
    AngularWidth,
    Cartesian1DiffWidth,
    Cartesian1DWidth,
)
from trackstream.utils.descriptors.classproperty import classproperty

__all__: list[str] = []

if TYPE_CHECKING:
    from astropy.units import PhysicalType

##############################################################################
# CODE
##############################################################################


@dataclass(frozen=True)
class Cartesian2DWidth(Cartesian1DWidth):
    """2D Cartesian width in configuration space."""

    y: Quantity  # [LENGTH]

    @classproperty
    def corresponding_representation_type(cls) -> type[coords.CartesianRepresentation]:
        """Return the corresponding representation type."""
        return coords.CartesianRepresentation

    @property
    def corresponding_width_types(self) -> dict[PhysicalType, None | type[BaseWidth]]:
        """Return the corresponding width types."""
        return {LENGTH: self.__class__, SPEED: Cartesian2DiffWidth}


@dataclass(frozen=True)
class PolarWidth(AngularWidth):
    """2D polar representation in configuration space."""

    distance: Quantity  # [LENGTH]

    @classproperty
    def corresponding_representation_type(cls) -> None:
        """Return the corresponding representation type."""
        return

    @property
    def corresponding_width_types(self) -> dict[PhysicalType, None | type[BaseWidth]]:
        """Return the corresponding width types."""
        return {LENGTH: self.__class__, SPEED: PolarDiffWidth}


# ===================================================================
# Kinematic Space


@dataclass(frozen=True)
class Cartesian2DiffWidth(Cartesian1DiffWidth):
    """3D Cartesian width in velocity space."""

    d_y: Quantity  # [SPEED]

    @property
    def corresponding_width_types(self) -> dict[PhysicalType, None | type[BaseWidth]]:
        """Return the corresponding width types."""
        return {LENGTH: None, SPEED: self.__class__}

    @classproperty
    def corresponding_representation_type(cls) -> None:
        """Return the corresponding representation type."""
        return


@dataclass(frozen=True)
class PolarDiffWidth(AngularDiffWidth):
    """3D-embedded 2D on-sky spherical width in velocity space."""

    d_distance: Quantity  # [SPEED]

    @classproperty
    def corresponding_representation_type(cls) -> None:
        """Return the corresponding representation type."""
        return

    @property
    def corresponding_width_types(self) -> dict[PhysicalType, None | type[BaseWidth]]:
        """Return the corresponding width types."""
        return {LENGTH: None, SPEED: self.__class__}
