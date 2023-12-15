"""Coordinates Utilities."""


from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Protocol, TypeVar

import astropy.coordinates as coords
import astropy.units as u
import numpy as np
from numpy import arcsin, arctan2, broadcast_to, cos, ndarray, pi, sin
import numpy.lib.recfunctions as rfn

from trackstream._typing import SupportsFrame
from trackstream.utils.coord_utils import deep_transform_to, f2q

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from astropy.coordinates import BaseCoordinateFrame, SkyCoord

__all__: list[str] = []

##############################################################################
# PARAMETERS

T = TypeVar("T")
R = TypeVar("R")  # return variable

_PI_2 = pi / 2

##############################################################################
# CODE
##############################################################################


@dataclass(frozen=True)
class FrameInfo(Generic[T]):
    """Frame information."""

    representation_type: type[coords.BaseRepresentation]
    differential_type: type[coords.BaseDifferential]
    units: u.StructuredUnit

    REGISTRY: ClassVar[MutableMapping[type, Any]] = {}

    def __set_name__(self, enclosing_cls: type[T], _: str) -> None:
        self.enclosing_cls: type[T]
        object.__setattr__(self, "enclosing_cls", enclosing_cls)

        # Register
        self.REGISTRY[self.enclosing_cls] = self

    def components(self, *, kinematics: bool) -> tuple[str, ...]:
        """Components.

        Parameters
        ----------
        kinematics : bool
            Whether to include the kinematic components.

        Returns
        -------
        tuple[str, ...]
            Names of components of representation and differential types.
        """
        cs = tuple(self.representation_type.attr_classes.keys())
        if kinematics:
            cs += tuple(self.differential_type.attr_classes.keys())
        return cs

    @property
    def dtype(self) -> np.dtype:
        """Dtype."""
        ldt = np.dtype([(n, float) for n in tuple(self.representation_type.attr_classes.keys())])
        sdt = np.dtype([(n, float) for n in tuple(self.differential_type.attr_classes.keys())])
        return np.dtype([("length", ldt), ("speed", sdt)])


class HasFrameAndInfo(SupportsFrame, Protocol):
    """Protocol for objects that have a frame and info."""

    @property
    def frame(self) -> BaseCoordinateFrame:
        """Frame."""
        ...

    @property
    def info(self) -> FrameInfo:
        """Frame info."""
        ...

    @property
    def nfeature(self) -> int:
        """Number of features."""
        ...


def _c2v(obj: HasFrameAndInfo, c: SkyCoord, /) -> ndarray:
    """Return unstructured array from coordinates.

    Parameters
    ----------
    obj : HasFrameAndInfo
        Object with frame and info.
    c : SkyCoord
        Coordinates.

    Returns
    -------
    ndarray
    """
    sc = deep_transform_to(c, obj.frame, obj.info.representation_type, obj.info.differential_type)
    return rfn.structured_to_unstructured(f2q(sc).to_value(obj.info.units))[:, : obj.nfeature]


def _v2c(obj: HasFrameAndInfo, arr: ndarray, /) -> BaseCoordinateFrame:
    # Position
    i: int = 0
    q: dict[str, u.Quantity] = {}
    for i, (n, acls) in enumerate(obj.info.representation_type.attr_classes.items()):
        v = arr[:, i] << obj.info.units[0][n]
        # Need to special case constrained classes
        if acls is coords.Latitude:  # map into (-90, 90)
            v = (((v.to_value(u.deg) + 90) % 180 - 90) << u.deg).to(v.unit)
        q[n] = v
    i += 1

    # The shape of arr determines if there are differentials!
    if arr.shape[1] > i:
        p = {}
        for j, n in enumerate(obj.info.differential_type.attr_classes.keys()):
            p[n] = arr[:, i + j] << obj.info.units[1][n]

        d = obj.info.differential_type(**p)
        r = obj.info.representation_type(**q, differentials={"s": d})
    else:
        r = obj.info.representation_type(**q)

    return obj.frame.realize_frame(
        r,
        representation_type=obj.info.representation_type,
        differential_type=obj.info.differential_type,
    )


##############################################################################


# -------------------------------------------------------------------
# Copied from Astropy, with units stripped. See License
# https://docs.astropy.org/en/stable/_modules/astropy/coordinates/angle_utilities.html
# The implementations in Astropy use Angle and are too slow.


def position_angle(lon1: ndarray, lat1: ndarray, lon2: float, lat2: float) -> ndarray:
    """Position Angle (East of North) between two points on a sphere.

    Parameters
    ----------
    lon1, lat1, lon2, lat2 : float['radian']
        |Longitude| and |Latitude| value of the two points in radians.

    Returns
    -------
    pa : float['radian']
        The (positive) position angle of the vector pointing from position 1 to
        position 2.  If any of the angles are arrays, this will contain an array
        following the appropriate `numpy` broadcasting rules.
    """
    deltalon = lon2 - lon1
    colat = cos(lat2)

    x = sin(lat2) * cos(lat1) - colat * sin(lat1) * cos(deltalon)
    y = sin(deltalon) * colat

    pa: ndarray = arctan2(y, x)
    return pa


def offset_by(lon: ndarray, lat: ndarray, posang: ndarray, distance: ndarray) -> tuple[ndarray, ndarray]:
    """Point with the given offset from the given point.

    Parameters
    ----------
    lon, lat, posang, distance : float['rad']
        |Longitude| and |Latitude| of the starting point,
        position angle and distance to the final point.
        Polar points at lat= ±90 are treated as limit of ±(90-epsilon) and same lon.

    Returns
    -------
    lon, lat : float['rad']
        The position of the final point.  If any of the angles are arrays,
        these will contain arrays following the appropriate `numpy` broadcasting rules.
        0 <= lon < 2pi.

    Notes
    -----
    See :mod:`astropy` implementation.
    """
    cos_a = cos(distance)
    sin_a = sin(distance)
    cos_c = sin(lat)
    sin_c = cos(lat)
    cos_B = cos(posang)
    sin_B = sin(posang)

    # cosine rule: Know two sides: a,c and included angle: B; get unknown side b
    cos_b = cos_c * cos_a + sin_c * sin_a * cos_B
    # sine rule and cosine rule for A (using both lets arctan2 pick quadrant).
    # multiplying both sin_A and cos_A by x=sin_b * sin_c prevents /0 errors
    # at poles.  Correct for the x=0 multiplication a few lines down.
    xsin_A = sin_a * sin_B * sin_c
    xcos_A = cos_a - cos_b * cos_c

    A = arctan2(xsin_A, xcos_A)  # radian
    # Treat the poles as if they are infinitesimally far from pole but at given lon
    small_sin_c = sin_c < 1e-12
    if small_sin_c.any():
        # For south pole (cos_c = -1), A = posang; for North pole, A=180 deg - posang
        A_pole = _PI_2 + cos_c * (_PI_2 - posang)
        if A.shape:
            # broadcast to ensure the shape is like that of A, which is also
            # affected by the (possible) shapes of lat, posang, and distance.
            small_sin_c = broadcast_to(small_sin_c, A.shape)
            A[small_sin_c] = A_pole[small_sin_c]
        else:
            A = A_pole

    outlon = lon + A
    outlat = arcsin(cos_b)

    return outlon, outlat
