# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------
#
# TITLE   :
# AUTHOR  :
# PROJECT :
#
# ----------------------------------------------------------------------------

"""Plot.

Description.

"""

__author__ = ""
# __copyright__ = "Copyright 2019, "
# __credits__ = [""]
# __license__ = ""
# __version__ = "0.0.0"
# __maintainer__ = ""
# __email__ = ""
# __status__ = "Production"


# __all__ = [
#     # functions
#     "",
#     # other
#     "",
# ]


##############################################################################
# IMPORTS

# BUILT-IN

# THIRD PARTY

import matplotlib.pyplot as plt

import numpy as np


# PROJECT-SPECIFIC


##############################################################################
# PARAMETERS


##############################################################################
# CODE
##############################################################################


def plot_SOM_visit_order(data, visit_order):
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
