"""Generate example data a la Palomar 5."""


##############################################################################
# IMPORTS

# STDLIB
import pathlib
from typing import Optional

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
from astropy.table import QTable
from astropy.units import Quantity
from astropy.utils.data import get_pkg_data_filename
from astropy.utils.misc import NumpyRNGContext
from galpy.orbit import Orbit
from galpy.potential import LogarithmicHaloPotential
from streamtools.df import streamspraydf

# LOCAL
from trackstream._typing import FullPathLike

__all__ = ["get_example_pal5"]

##############################################################################
# PARAMETERS

DIR = pathlib.Path(__file__).parent

##############################################################################
# CODE
##############################################################################


def make_stream_from_Vasiliev18(
    name: str, tdisrupt: Quantity = 5 * u.Gyr, *, write: Optional[FullPathLike] = None
) -> QTable:
    """Make and write data table.

    Parameters
    ----------
    name : str
    tdisrupt : `~astropy.units.Quantity`
    """

    # read data
    table = QTable.read(get_pkg_data_filename("Vasiliev18_table.ecsv", package="trackstream.example_data"))
    table.add_index("Name")

    # get the Pal-5 subset
    stable = table.loc[name]

    # get the origin -- the GC
    sgc = coord.SkyCoord(
        ra=stable["ra"],
        dec=stable["dec"],
        distance=stable["dist"],
        pm_ra_cosdec=stable["pmra"],
        pm_dec=stable["pmdec"],
        radial_velocity=stable["vlos"],
    )

    # Create a Potential
    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    lp.turn_physical_on()

    # progenitor properties
    o = Orbit(sgc)
    mass = 2 * 10.0**4.0 * u.Msun
    # tdisrupt = 5 * u.Gyr

    ro, vo = 8 * u.kpc, 220 * u.km / u.s

    # Streamspray of the tidal arms
    # leading
    spdf_l = streamspraydf(progenitor_mass=mass, progenitor=o, pot=lp, tdisrupt=tdisrupt, leading=True, ro=ro, vo=vo)

    # trailing
    spdf_t = streamspraydf(progenitor_mass=mass, progenitor=o, pot=lp, tdisrupt=tdisrupt, leading=False, ro=ro, vo=vo)

    # make sample
    with NumpyRNGContext(4):

        RvR_l, dt = spdf_l.sample(n=300, returndt=True, integrate=True)
        RvR_t, dt = spdf_t.sample(n=300, returndt=True, integrate=True)

    # get coordinates
    data_l = Orbit(RvR_l.T, ro=ro, vo=vo).SkyCoord()
    data_t = Orbit(RvR_t.T, ro=ro, vo=vo).SkyCoord()

    # turn into QTable
    data = QTable(dict(coord=coord.concatenate((data_l, data_t))))
    data["x_err"] = 0 * u.kpc
    data["y_err"] = 0 * u.kpc
    data["z_err"] = 0 * u.kpc
    data["Pmemb"] = 100 * u.percent
    data["tail"] = (["arm1"] * len(data_l)) + (["arm2"] * len(data_t))

    # add some metadata
    data.meta["origin"] = sgc

    # save data
    if write is not None:
        data.write(write, overwrite=True)

    return data


def get_example_stream(name: str) -> QTable:
    try:
        fname = get_pkg_data_filename(f"example_data/{name.lower()}_ex.ecsv", package="trackstream")
    except Exception as e:
        print(e)
        data = make_stream_from_Vasiliev18(name=name, write=DIR / f"{name.lower()}_ex.ecsv")
    else:
        data = QTable.read(fname)

    return data


def get_example_pal5() -> QTable:
    return get_example_stream("Pal_5")


##############################################################################


if __name__ == "__main__":
    get_example_pal5()
