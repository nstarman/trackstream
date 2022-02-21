# -*- coding: utf-8 -*-

"""Plot Preprocessing."""


__all__ = [
    # functions
    "plot_rotation_frame_residual",
]


##############################################################################
# IMPORTS

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import imshow_norm

# LOCAL
from .rotated_frame import residual as fit_rotated_frame_residual

##############################################################################
# CODE
##############################################################################


def plot_rotation_frame_residual(
    data,
    origin: coord.BaseCoordinateFrame,
    num_rots: int = 3600,
    scalar: bool = True,
) -> plt.Figure:
    """Plot residual from finding the optimal rotated frame.

    Parameters
    ----------
    data : Coordinate
    origin : ICRS
    num_rots : int, optional
        Number of rotation angles in (-180, 180) to plot.
    scalar : bool, optional
        Whether to plot scalar or full vector residual.

    Returns
    -------
    `~matplotlib.pyplot.Figure`
    """
    # Get data
    frame = data.replicate_without_data()
    origin = origin.transform_to(frame).represent_as(coord.SphericalRepresentation)
    lon = origin.lon.to_value(u.deg)
    lat = origin.lat.to_value(u.deg)

    # Evaluate residual
    rs = np.linspace(-180, 180, num=num_rots)
    res = np.array(
        [
            fit_rotated_frame_residual(
                (r, lon, lat),
                data=data.represent_as(coord.CartesianRepresentation),
                scalar=scalar,
            )
            for r in rs
        ],
    )

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    if scalar:
        ax.scatter(rs, res)
        ax.set_xlabel(r"Rotation angle $\theta$")
        ax.set_ylabel(r"residual")

    else:
        im, norm = imshow_norm(res.T, ax=ax, aspect="auto", origin="lower")
        # xticks
        locs = ax.get_xticks()
        xticks = [str(int(loc // 10)) for loc in locs]
        plt.xticks(locs[1:-1], xticks[1:-1])
        # labels
        ax.set_xlabel(r"$\theta$ + 180 [deg]")
        ax.set_ylabel(r"phi2")

        # colorbar
        cbar = fig.colorbar(im)
        cbar.ax.set_ylabel("residual")

    return fig


# -------------------------------------------------------------------


def plot_SOM(data, order):
    """Plot SOM.

    Parameters
    ----------
    data
    order

    returns

    """
    fig, ax = plt.subplots(figsize=(10, 9))

    pts = ax.scatter(
        data[order, 0],
        data[order, 1],
        c=np.arange(0, len(data)),
        vmax=len(data),
        cmap="plasma",
        label="data",
    )

    ax.plot(data[order][:, 0], data[order][:, 1], c="gray")

    cbar = plt.colorbar(pts, ax=ax)
    cbar.ax.set_ylabel("SOM ordering")

    fig.legend(loc="upper left")
    fig.tight_layout()

    return fig


# -------------------------------------------------------------------

##############################################################################
# END
