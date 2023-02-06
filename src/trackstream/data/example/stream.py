"""Generate example Palomar 5 data."""

##############################################################################
# IMPORTS

# STDLIB
import os
import pathlib
from typing import cast

# THIRD PARTY
import astropy.coordinates as coords
import astropy.units as u
import numpy as np
from astropy.table import QTable
from astropy.units import Quantity
from astropy.utils.data import get_pkg_data_filename
from astropy.utils.misc import NumpyRNGContext
from galpy.df import streamspraydf
from galpy.orbit import Orbit
from galpy.potential import LogarithmicHaloPotential

__all__ = ["get_example_stream"]


##############################################################################
# PARAMETERS

DIR = pathlib.Path(__file__).parent
FullPathLike = str | bytes | os.PathLike

##############################################################################
# CODE
##############################################################################


def make_stream_from_vasiliev18(
    name: str,
    tdisrupt: Quantity = Quantity(5, u.Gyr),
    *,
    write: FullPathLike | None = None,
) -> QTable:
    """Make and write data table.

    Parameters
    ----------
    name : str
        The name of the stream.
    tdisrupt : `~astropy.units.Quantity`
        The time at which the stream is disrupted.
    write : str | bytes | `os.PathLike` | None
        If not `None`, write the data table to this path.

    Returns
    -------
    `astropy.table.QTable`
    """
    # Tead data
    fname = get_pkg_data_filename("Vasiliev18.ecsv", package="trackstream.data")
    table = QTable.read(fname)
    table.add_index("Name")

    # Get the Pal-5 subset
    stable = table.loc[name]

    # Get the origin -- the GC
    sgc = coords.SkyCoord(
        ra=stable["ra"],
        dec=stable["dec"],
        distance=stable["dist"],
        pm_ra_cosdec=stable["pmra"],
        pm_dec=stable["pmdec"],
        radial_velocity=stable["vlos"],
    )
    o = Orbit(sgc)

    # Create a Potential
    lp = LogarithmicHaloPotential(normalize=True, q=0.9)
    lp.turn_physical_on()

    # progenitor properties
    mass = Quantity(2 * 10.0**4.0, u.Msun)
    mass = Quantity(2 * 10.0**4.0, u.Msun)

    ro = Quantity(8, u.kpc)
    vo = Quantity(220, u.km / u.s)

    # Streamspray of the tidal arms : leading & trailing
    spdf_l = streamspraydf(progenitor_mass=mass, progenitor=o, pot=lp, tdisrupt=tdisrupt, leading=True, ro=ro, vo=vo)
    spdf_t = streamspraydf(progenitor_mass=mass, progenitor=o, pot=lp, tdisrupt=tdisrupt, leading=False, ro=ro, vo=vo)

    # Sample from leading and trailing
    with NumpyRNGContext(4):
        RvR_l, *_ = spdf_l.sample(n=300, returndt=True, integrate=True)
        RvR_t, *_ = spdf_t.sample(n=300, returndt=True, integrate=True)

    RvR_l = cast("np.ndarray", RvR_l)
    RvR_t = cast("np.ndarray", RvR_t)

    # Get coordinates
    data_l = Orbit(RvR_l.T, ro=ro, vo=vo).SkyCoord()
    data_t = Orbit(RvR_t.T, ro=ro, vo=vo).SkyCoord()

    # Turn into QTable
    data = QTable({"coord": coords.concatenate((data_l, data_t))})
    data["x_err"] = Quantity(0, u.kpc)
    data["y_err"] = Quantity(0, u.kpc)
    data["z_err"] = Quantity(0, u.kpc)
    data["Pmemb"] = Quantity(100, u.percent)
    data["arm"] = (["arm1"] * len(data_l)) + (["arm2"] * len(data_t))

    # Add some metadata
    data.meta["origin"] = sgc

    # save data
    if write is not None:
        data.write(write, overwrite=True)

    return data


def get_example_stream(name: str) -> QTable:
    """Get example stream.

    Parameters
    ----------
    name : {'Pal_5', 'Pal_11'}
        The (file) name of the stream.

    Returns
    -------
    `astropy.table.QTable`
        Data table.
    """
    try:
        fname = get_pkg_data_filename(f"example_data/{name}_ex.ecsv", package="trackstream")
    except OSError:
        data = make_stream_from_vasiliev18(name=name, write=DIR / f"{name}_ex.ecsv")
    else:
        data = QTable.read(fname)

    return data
