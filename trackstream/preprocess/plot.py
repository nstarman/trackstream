# -*- coding: utf-8 -*-

"""Plot Preprocessing."""


__all__ = [
    # functions
    "plot_rotation_frame_residual",
]


##############################################################################
# IMPORTS

# BUILT-IN

# THIRD PARTY

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import imshow_norm

from .rotated_frame import residual as fit_rotated_frame_residual

# PROJECT-SPECIFIC


##############################################################################
# PARAMETERS


##############################################################################
# CODE
##############################################################################


def plot_rotation_frame_residual(
    data, origin: coord.BaseCoordinateFrame, num_rots: int = 3600, scalar: bool = True,
) -> plt.Figure:
    """Plot residual from rotation frame.

    Parameters
    ----------
    data : Coordinate
    origin : ICRS
    num_rots : int
        Number of rotation angles in (-180, 180) to plot.
    scalar : bool
        Whether to plot scalar or full vector residual.

    Returns
    -------
    `~matplotlib.pyplot.Figure`

    """
    origin = origin.transform_to(data.__class__).represent_as(
        coord.SphericalRepresentation
    )
    lon = origin.lon.to_value(u.deg)
    lat = origin.lat.to_value(u.deg)

    rs = np.linspace(-180, 180, num=num_rots)
    res = np.array(
        [
            fit_rotated_frame_residual(
                (r, lon, lat),
                data=data.represent_as(coord.CartesianRepresentation),
                scalar=scalar,
            )
            for r in rs
        ]
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    if scalar:

        ax.scatter(rs, res)
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r"residual")

    else:

        im, norm = imshow_norm(res.T, ax=ax, aspect="auto", origin="lower")
        ax.set_xlabel(r"$\theta$/10 + 180 [deg]")
        ax.set_ylabel(r"phi2")

        cbar = fig.colorbar(im)
        cbar.ax.set_ylabel("residual")

    return fig


# /def


# -------------------------------------------------------------------


##############################################################################
# END
