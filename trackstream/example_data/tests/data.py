# -*- coding: utf-8 -*-

"""Example Data.

.. todo::

    Use the pytest fixtures method for multi-module data.

"""

# __all__ = [
# ]


##############################################################################
# IMPORTS

import numpy as np
import astropy.units as u
import astropy.coordinates as coord
from trackstream.example_data import example_coords

##############################################################################
# DATA
##############################################################################

ricrs = example_coords.RotatedICRS(
    phi1=np.linspace(-np.pi, np.pi, 128) * u.radian,
    phi2=np.zeros(128) * u.radian,
    pm_phi1_cosphi2=np.ones(128) * 10 * u.mas / u.yr,
    pm_phi2=np.zeros(128) * u.mas / u.yr,
)

icrs = ricrs.transform_to(coord.ICRS)


# -------------------------------------------------------------------

rgcentric = example_coords.RotatedGalactocentric(
    phi1=np.linspace(-np.pi, np.pi, 128) * u.radian,
    phi2=np.zeros(128) * u.radian,
    pm_phi1_cosphi2=np.ones(128) * 10 * u.mas / u.yr,
    pm_phi2=np.zeros(128) * u.mas / u.yr,
)

gcentric = rgcentric.transform_to(coord.Galactocentric)
gcentric.representation_type = coord.SphericalRepresentation


##############################################################################
# END
