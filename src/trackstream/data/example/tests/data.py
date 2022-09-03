"""Tests for example data."""

##############################################################################
# IMPORTS

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import numpy as np

# LOCAL
from trackstream.data.example.frame import RotatedGalactocentric, RotatedICRS

__all__: list[str] = []

##############################################################################
# DATA
##############################################################################

# -------------------------------------------------------------------
# Rotated ICRS data


ricrs = RotatedICRS(
    phi1=np.linspace(-np.pi, np.pi, 128) * u.radian,
    phi2=u.Quantity(np.zeros(128), u.radian),
    pm_phi1_cosphi2=u.Quantity(np.ones(128) * 10, u.Unit("mas / yr")),
    pm_phi2=u.Quantity(np.zeros(128), u.Unit("mas / yr")),
)

icrs = ricrs.transform_to(coord.ICRS)


# -------------------------------------------------------------------
# Rotated Galactocentric data

rgcentric = RotatedGalactocentric(
    phi1=u.Quantity(np.linspace(-np.pi, np.pi, 128), u.radian),
    phi2=u.Quantity(np.zeros(128), u.radian),
    pm_phi1_cosphi2=u.Quantity(np.ones(128) * 10, u.Unit("mas / yr")),
    pm_phi2=u.Quantity(np.zeros(128), u.Unit("mas / yr")),
)

gcentric = rgcentric.transform_to(coord.Galactocentric)
gcentric.representation_type = coord.SphericalRepresentation
