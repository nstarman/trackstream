"""Widths."""

from trackstream.track.width.core import BaseWidth
from trackstream.track.width.interpolated import InterpolatedWidth, InterpolatedWidths
from trackstream.track.width.oned.core import (
    AngularDiffWidth,
    AngularWidth,
    Cartesian1DiffWidth,
    Cartesian1DWidth,
)
from trackstream.track.width.plural import Widths
from trackstream.track.width.threed.core import (
    Cartesian3DiffWidth,
    Cartesian3DWidth,
    SphericalDiffWidth,
    SphericalWidth,
    UnitSphericalDiffWidth,
    UnitSphericalWidth,
)
from trackstream.track.width.transforms import represent_as
from trackstream.track.width.twod.core import (
    Cartesian2DiffWidth,
    Cartesian2DWidth,
    PolarDiffWidth,
    PolarWidth,
)

# import to register interoperations
# isort: split

from trackstream.track.width import interop  # noqa: F401

__all__ = [
    # Widths
    "BaseWidth",
    # 1d
    "AngularWidth",
    "AngularDiffWidth",
    "Cartesian1DWidth",
    "Cartesian1DiffWidth",
    # 2d,
    "Cartesian2DWidth",
    "PolarWidth",
    "Cartesian2DiffWidth",
    "PolarDiffWidth",
    # 3d
    "Cartesian3DWidth",
    "UnitSphericalWidth",
    "SphericalWidth",
    "Cartesian3DiffWidth",
    "SphericalDiffWidth",
    "UnitSphericalDiffWidth",
    # Bundle
    "Widths",
    # Wrappers
    "InterpolatedWidth",
    "InterpolatedWidths",
    # functions
    "represent_as",
]
