"""Example Orbit Data."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from typing import TYPE_CHECKING, Union, cast

# THIRD PARTY
import astropy.coordinates as coords
import astropy.units as u
import numpy as np
from galpy import potential
from galpy.orbit import Orbit
from numpy.random import Generator, RandomState, default_rng

# LOCAL
from trackstream._typing import CoordinateType

if TYPE_CHECKING:
    # THIRD PARTY
    from typing_extensions import TypeAlias


__all__ = [
    "get_orbit",
    "make_ordered_orbit_data",
    "make_unordered_orbit_data",
    "make_noisy_orbit_data",
]
__author__ = "Nathaniel Starkman"
__credits__ = ["Jo Bovy"]


##############################################################################
# PARAMETERS

stop: int = 200
num: int = 100
unit = u.Unit("Myr")

UnitType: TypeAlias = Union[u.Unit, u.IrreducibleUnit, u.UnitBase, u.FunctionUnitBase]
CoordinateLikeType: TypeAlias = Union[CoordinateType, str]
RepresentationLikeType: TypeAlias = Union[coords.BaseRepresentation, str]

##############################################################################
# CODE
##############################################################################


def make_shuffler(length: int, rng: Generator | RandomState | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Shuffle and un-shuffle arrays.

    Parameters
    ----------
    length : int
        Array length for which to construct (un)shuffle arrays.
    rng : `~numpy.random.Generator`, optional
        random number generator.

    Returns
    -------
    shuffler : `~numpy.ndarray`
        index array that shuffles any array of size `length` along
        a specified axis
    undo : `~numpy.ndarray`
        index array that undoes above, if applied identically.
    """
    if rng is None:
        rng = default_rng()

    shuffler = np.arange(length)  # start with index array
    rng.shuffle(shuffler)  # shuffle array in-place

    undo = shuffler.argsort()  # and construct the un-shuffler

    return shuffler, undo


def get_orbit(stop: float = stop, num: int = num, unit: UnitType = unit) -> Orbit:
    """Get Orbit by integrating in a `galpy.potential.MWPotential2014`.

    Parameters
    ----------
    stop: float
    num : int
    unit : `~astropy.units.Unit`

    Returns
    -------
    rep: `~galpy.orbit.Orbit`
        Integrated orbit.
    """
    # create time integration array
    time = np.linspace(0, stop, num=num) * unit

    # integrate orbit
    o = Orbit()
    o.integrate(time, potential.MWPotential2014)

    return o


# -------------------------------------------------------------------


def make_ordered_orbit_data(
    stop: float = stop,
    num: int = num,
    unit: UnitType = unit,
    frame: CoordinateLikeType = "galactocentric",
    representation_type: RepresentationLikeType = "cartesian",
) -> coords.SkyCoord:
    """Make Ordered Orbit Data.

    Parameters
    ----------
    stop : float
    num : int
    unit : Unit

    Returns
    -------
    SkyCoord
        (`num`, 3) array
    """
    # Get orbit
    time = np.linspace(0, stop, num=num) * unit
    o = Orbit()
    o.integrate(time, potential.MWPotential2014)

    # Extract coordinates in correct frame
    sc = o.SkyCoord(o.time())
    tsc = sc.transform_to(frame)
    tsc.representation_type = representation_type

    return tsc


# -------------------------------------------------------------------


def make_unordered_orbit_data(
    stop: float = stop,
    num: int = num,
    unit: UnitType = unit,
    frame: CoordinateLikeType = "galactocentric",
    representation_type: RepresentationLikeType = "cartesian",
) -> coords.SkyCoord:
    """Make Ordered Orbit Data.

    Parameters
    ----------
    stop : float
    num : int
    unit : Unit

    Returns
    -------
    SkyCoord
        (`num`, 3) array
    """
    # Ordered data
    osc = make_ordered_orbit_data(stop=stop, num=num, unit=unit, frame=frame, representation_type=representation_type)
    # Shuffle the data
    shuffler, _ = make_shuffler(len(osc))
    usc = cast(coords.SkyCoord, osc[shuffler])
    return usc


# -------------------------------------------------------------------


def make_noisy_orbit_data(
    stop: float = stop,
    num: int = num,
    sigma: dict[str, u.Quantity] | None = None,
    unit: UnitType = unit,
    frame: CoordinateLikeType = "galactocentric",
    representation_type: RepresentationLikeType = "cartesian",
    rnd: int | np.random.Generator | None = None,
) -> coords.SkyCoord:
    """Make ordered orbit data.

    Parameters
    ----------
    stop : float
    num : int
    sigma : dict[str, Quantity] or None, optional
        Errors in Galactocentric Cartesian coordinates.
    unit : Unit

    Returns
    -------
    SkyCoord
        (`num`, 3) array
    """
    # Get random state
    rnd = rnd if isinstance(rnd, Generator) else default_rng(seed=rnd)

    # Default error
    if sigma is None:
        sig = dict(x=u.Quantity(100, u.pc), y=u.Quantity(100, u.pc), z=u.Quantity(20, u.pc))
    else:
        sig = sigma

    # Unordered data
    usc = make_unordered_orbit_data(
        stop=stop, num=num, unit=unit, frame="galactocentric", representation_type="cartesian"
    )

    # Noisy SkyCoord with gaussian-convolved values.
    noisy: dict[str, u.Quantity] = {}
    for n, unit in cast(u.StructuredUnit, cast(u.Quantity, usc.data)._units).items():
        mean = getattr(usc.data, n).to_value(unit)
        scale = cast(np.ndarray, sig[n].to_value(unit))
        noisy[n] = u.Quantity(rnd.normal(mean, scale=scale), unit=unit)

    nc = coords.SkyCoord(usc.frame.realize_frame(coords.CartesianRepresentation(**noisy))).transform_to(frame)
    nc.representation_type = representation_type

    return coords.SkyCoord(nc)
