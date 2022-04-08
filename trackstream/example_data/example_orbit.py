# -*- coding: utf-8 -*-

"""Example Orbit Data."""

__author__ = "Nathaniel Starkman"
__credits__ = ["Jo Bovy"]


__all__ = [
    "get_orbit",
    "make_ordered_orbit_data",
    "make_unordered_orbit_data",
    "make_noisy_orbit_data",
]


##############################################################################
# IMPORTS

# STDLIB
from typing import Dict, Optional, Union

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from galpy import potential
from galpy.orbit import Orbit

# LOCAL
from trackstream._type_hints import FrameLikeType, RepresentationLikeType, UnitType
from trackstream.utils.misc import make_shuffler

##############################################################################
# PARAMETERS

stop = 200
num = 100
unit = u.Myr

##############################################################################
# CODE
##############################################################################


def get_orbit(stop: float = stop, num: int = num, unit: UnitType = unit) -> Orbit:
    """Get Orbit.

    Parameters
    ----------
    stop: float
    num : int
    unit : `~astropy.units.Unit`

    Returns
    -------
    rep: `~galpy.orbit.Orbit`
        Integrated Orbit
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
    frame: FrameLikeType = "galactocentric",
    representation_type: RepresentationLikeType = "cartesian",
) -> SkyCoord:
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
    o = get_orbit(stop=stop, num=num, unit=unit)
    sc = o.SkyCoord(o.time())

    tsc = sc.transform_to(frame)
    tsc.representation_type = representation_type

    return tsc


# -------------------------------------------------------------------


def make_unordered_orbit_data(
    stop: float = stop,
    num: int = num,
    unit: UnitType = unit,
    frame: FrameLikeType = "galactocentric",
    representation_type: RepresentationLikeType = "cartesian",
) -> coord.BaseRepresentation:
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
    osc = make_ordered_orbit_data(
        stop=stop,
        num=num,
        unit=unit,
        frame=frame,
        representation_type=representation_type,
    )

    shuffler, _ = make_shuffler(len(osc))
    usc = osc[shuffler]

    return usc


# -------------------------------------------------------------------


def make_noisy_orbit_data(
    stop: float = stop,
    num: int = num,
    sigma: Optional[Dict[str, u.Quantity]] = None,
    unit: UnitType = unit,
    frame: FrameLikeType = "galactocentric",
    representation_type: RepresentationLikeType = "cartesian",
    rnd: Union[int, np.random.Generator, None] = None,
) -> SkyCoord:
    """Make Ordered Orbit Data.

    Parameters
    ----------
    stop : float
    num : int
    sigma : dict[str, Quantity] or None, optional
        Errors in Galactocentric Cartesian coordinates
    unit : Unit

    Returns
    -------
    SkyCoord
        (`num`, 3) array

    """
    # Get random state
    rnd = rnd if isinstance(rnd, np.random.Generator) else np.random.default_rng(seed=rnd)

    if sigma is None:
        sigma = dict(x=100 * u.pc, y=100 * u.pc, z=20 * u.pc)

    usc = make_unordered_orbit_data(
        stop=stop,
        num=num,
        unit=unit,
        frame="galactocentric",
        representation_type="cartesian",
    )

    # Noisy SkyCoord with gaussian-convolved values.
    noisy = {
        n: rnd.normal(getattr(usc.data, n).to_value(unit), scale=sigma[n].to_value(unit)) * unit
        for n, unit in usc.data._units.items()
    }
    nsc = SkyCoord(usc.realize_frame(coord.CartesianRepresentation(**noisy)))

    # transformed to desired frame and representation type
    sc = nsc.transform_to(frame)
    sc.representation_type = representation_type

    return sc
