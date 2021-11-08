# -*- coding: utf-8 -*-

"""Plot."""


__all__ = [
    "plot_dts",
    "plot_path",
]


##############################################################################
# IMPORTS

# THIRD PARTY
import matplotlib.pyplot as plt
import numpy as np
from filterpy.stats import plot_covariance  # TODO replace

##############################################################################
# PARAMETERS


##############################################################################
# CODE
##############################################################################


def plot_dts(distances, averaged_distances):
    """Plot time arrays.

    Parameters
    ----------
    distances
    averaged_distances

    Returns
    -------
    `~matplotlib.pyplot.Figure`

    """
    dts = np.arange(len(distances))  # indices

    fig, ax = plt.subplots()

    ax.scatter(dts, distances, marker="*", c="k", label="p2p distance")

    c = np.arange(0, len(dts))
    ax.scatter(dts, averaged_distances, c=c, cmap="plasma_r", label="smoothed")

    ax.set_xlabel("index")
    ax.set_ylabel("distance")
    ax.set_title("point to point distance")

    plt.legend()
    plt.tight_layout()

    return fig


# /def


# -------------------------------------------------------------------


def plot_path(data, path, cov=None, true_path=None, *, num_std=1, cov_alpha=0.5, is_ordered=False):
    """Plot time arrays.

    Parameters
    ----------
    data : array-like
    path : array-like
    cov : array-like, optional
        Default None
    true_path : array-like, optional
        Default None

    num_std : int, optional, keyword only
        Default 1
    cov_alpha : float, optional, keyword only
        Default 0.5
    is_ordered : bool, optional, keyword only
        Default False

    Returns
    -------
    `~matplotlib.pyplot.Figure`
    `~matplotlib.pyplot.Axes`

    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    axs[0].set_title("x, y")
    axs[0].scatter(data.x, data.y)
    if is_ordered:
        axs[0].scatter(data.x[0], data.y[0], c="y", s=100)
    if true_path is not None:
        axs[0].plot(true_path.x, true_path.y, c="g")
    if cov is not None:
        plt.sca(axs[0])
        for i, p in enumerate(cov):
            P = np.array([[p[0, 0], p[2, 0]], [p[0, 2], p[2, 2]]])
            mean = (path.x[i], path.y[i])
            plot_covariance(
                mean,
                cov=P,
                fc="gray",
                std=num_std,
                alpha=cov_alpha,
            )
        axs[0].set_aspect("auto")
    axs[0].plot(path.x, path.y, c="k")

    axs[1].set_title("x, z")
    axs[1].scatter(data.x, data.z)
    if is_ordered:
        axs[1].scatter(data.x[0], data.z[0], c="y", s=100)
    if true_path is not None:
        axs[1].plot(true_path.x, true_path.z, c="g")

    if cov is not None:
        plt.sca(axs[1])
        for i, p in enumerate(cov):
            P = np.array([[p[0, 0], p[4, 0]], [p[0, 4], p[4, 4]]])
            mean = (path.x[i], path.z[i])
            plot_covariance(
                mean,
                cov=P,
                fc="gray",
                std=num_std,
                alpha=cov_alpha,
            )
        axs[1].set_aspect("auto")
    axs[1].plot(path.x, path.z, c="k")

    axs[2].set_title("y, z")
    axs[2].scatter(data.y, data.z)
    if is_ordered:
        axs[2].scatter(data.y[0], data.z[0], c="y", s=100)
    if true_path is not None:
        axs[2].plot(true_path.y, true_path.z, c="g")
    if cov is not None:
        plt.sca(axs[2])
        for i, p in enumerate(cov):
            P = np.array([[p[2, 2], p[4, 2]], [p[2, 4], p[4, 4]]])
            mean = (path.y[i], path.z[i])
            plot_covariance(
                mean,
                cov=P,
                fc="gray",
                std=num_std,
                alpha=cov_alpha,
            )
        axs[2].set_aspect("auto")
    axs[2].plot(path.y, path.z, c="k")

    plt.tight_layout()

    return fig, axs


# /def

##############################################################################
# END
