# -*- coding: utf-8 -*-

"""Example Orbit Data."""

__author__ = "Nathaniel Starkman"
__credits__ = ["Jo Bovy"]


__all__ = [
    # functions
    "get_orbit",
    "make_ordered_orbit_data",
    "make_unordered_orbit_data",
    "make_noisy_orbit_data",
]


##############################################################################
# IMPORTS

# STDLIB
import typing as T

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
from galpy import potential
from galpy.orbit import Orbit

# LOCAL
from trackstream._type_hints import FrameLikeType, RepLikeType, UnitType
from trackstream.utils.misc import make_shuffler

##############################################################################
# PARAMETERS

stop = 200
num = 100
unit = u.Myr

##############################################################################
# CODE
##############################################################################


def get_orbit(stop: float = stop, num: int = num, unit: UnitType = unit):
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
    representation_type: RepLikeType = "cartesian",
) -> coord.BaseRepresentation:
    """Make Ordered Orbit Data.

    Parameters
    ----------
    stop : float
    num : int
    unit : Unit

    Returns
    -------
    data : Sequence
        (`num`, 3) array

    """
    o = get_orbit(
        stop=stop,
        num=num,
        unit=unit,
    )

    # get representation
    sc = o.SkyCoord(o.time())
    sc_new = sc.transform_to(frame)

    rep: coord.BaseRepresentation = sc_new.represent_as(representation_type)

    data = rep.without_differentials()
    # data = data.get_xyz().T

    return data


# -------------------------------------------------------------------


def make_unordered_orbit_data(
    stop: float = stop,
    num: int = num,
    unit: UnitType = unit,
    frame: FrameLikeType = "galactocentric",
    representation_type: RepLikeType = "cartesian",
) -> coord.BaseRepresentation:
    """Make Ordered Orbit Data.

    Parameters
    ----------
    stop : float
    num : int
    unit : Unit

    Returns
    -------
    data : Sequence
        (`num`, 3) array

    """
    X = make_ordered_orbit_data(
        stop=stop,
        num=num,
        unit=unit,
        frame=frame,
        representation_type=representation_type,
    )

    shuffler, undo = make_shuffler(len(X))

    data = X[shuffler]

    return data


# -------------------------------------------------------------------


def make_noisy_orbit_data(
    stop: float = stop,
    num: int = num,
    sigma: T.Optional[T.Dict[str, float]] = None,
    unit: UnitType = unit,
    frame: FrameLikeType = "galactocentric",
    representation_type: RepLikeType = "cartesian",
    rnd=None,
):
    """Make Ordered Orbit Data.

    Parameters
    ----------
    stop : float
    num : int
    sigma : tuple
        (0.35, 0.35, 0.08)
    unit : Unit

    Returns
    -------
    data : Sequence
        (`num`, 3) array

    """
    if rnd is None:
        try:
            rnd = np.random.default_rng(seed=None)
        except AttributeError:
            rnd = np.random.RandomState(seed=None)

    if sigma is None:
        if representation_type.lower() == "cartesian":
            sigma = dict(x=0.25, y=0.25, z=0.01)
        else:
            raise ValueError(
                (
                    "don't have a default sigma for "
                    "representation_type = {}".format(representation_type)
                ),
            )

    X = make_unordered_orbit_data(
        stop=stop,
        num=num,
        unit=unit,
        frame=frame,
        representation_type=representation_type,
    )

    recarr = X._values  # numpy recarray

    # make representation with gaussian-convolved values.
    data = X.__class__(**{n: rnd.normal(recarr[n], scale=sigma[n]) for n in recarr.dtype.names})

    return data


# -------------------------------------------------------------------

##############################################################################
# END
