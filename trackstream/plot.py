# -*- coding: utf-8 -*-

"""Plot functions."""

__all__ = [
    "plot_SOM",
]


##############################################################################
# IMPORTS

# THIRD PARTY
import matplotlib.pyplot as plt
import numpy as np

##############################################################################
# PARAMETERS


##############################################################################
# CODE
##############################################################################


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


# /def

# -------------------------------------------------------------------


##############################################################################
# END
