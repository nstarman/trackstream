# -*- coding: utf-8 -*-

"""Plot Functions."""

__all__ = [
    "plot_SOM_visit_order",
]


##############################################################################
# IMPORTS

# THIRD PARTY
import matplotlib.pyplot as plt
import numpy as np

##############################################################################
# CODE
##############################################################################


def plot_SOM_visit_order(data, visit_order) -> plt.Figure:
    """Plot the SOM visit order.

    Parameters
    ----------
    data
    visit_order

    Returns
    -------
    fig : `~matplotlib.pyplot.Figure`

    """
    x = np.arange(len(visit_order))

    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    axs[0].scatter(x, data[visit_order, 0])
    axs[0].set_xlabel("index")
    axs[0].set_ylabel("X")

    axs[1].scatter(x, data[visit_order, 1])
    axs[1].set_xlabel("index")
    axs[1].set_ylabel("Y")

    axs[2].scatter(x, data[visit_order, 2])
    axs[2].set_xlabel("index")
    axs[2].set_ylabel("Z")

    plt.tight_layout()

    return fig


# /def

# -------------------------------------------------------------------


##############################################################################
# END
