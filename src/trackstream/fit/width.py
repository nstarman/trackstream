"""Stream width."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from typing import TYPE_CHECKING

# THIRD PARTY
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.units import Quantity

# LOCAL
from trackstream.fit.som import SelfOrganizingMap1DBase

if TYPE_CHECKING:
    # LOCAL
    from trackstream.stream.core import StreamArm  # noqa: F401


##############################################################################
# CODE
##############################################################################


def make_stream_width(data: SkyCoord, som: SelfOrganizingMap1DBase, width0: Quantity) -> Quantity:
    """Make stream width

    Parameters
    ----------
    data : SkyCoord
        Ordered.
    """
    width = Quantity(np.ones((len(data),), dtype=width0.dtype), width0.unit)

    x = som._crd_to_v(data)
    _, _, _, distances = som._get_info_for_projection(x)

    # find the index of the best orthogonal distance (including nodes)
    ind_best_distance = np.argmin(distances, axis=1)
    orth_distance = distances[(np.arange(len(distances))), (ind_best_distance)]

    # TODO! better kernel width selection
    Ds = np.convolve(orth_distance, np.ones((10,)) / 10, mode="same")

    # TODO! more robust assignment and making sure of units
    for k in width.unit.keys():
        width[k] = width0[k]  # broadcasts
        if k in som.units.field_names:
            d = Ds * som.units[k]
            sel = width[k] < d
            width[k][sel] = d[sel]

    return width
