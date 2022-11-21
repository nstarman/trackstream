"""Utilities for SOMs."""

from __future__ import annotations

# STDLIB
from math import pi
from typing import TypeVar

# THIRD PARTY
import numpy as np
from numpy import arccos, cos, diff, ndarray, nonzero
from numpy.linalg import norm

__all__: list[str] = []

##############################################################################
# PARAMETERS


NDT = TypeVar("NDT", bound=ndarray)


##############################################################################
# CODE
##############################################################################


def _respace_bins_from_left(bins: NDT, maxsep: ndarray, onsky: bool, eps: float | np.floating) -> NDT:
    """Respace bins to have a maximum separation.

    Bins are respaced from the left-most bin up to the penultimate bin. The
    respaced bins are iteratively processed until all bins (except the last) are
    no more than ``maxsep`` separated.

    Parameters
    ----------
    bins : (N,) ndarray
        The bins to make sure are correctly spaced. Must be sorted.
    maxsep : scalar ndarray
        Maximum separation between bins.
    onsky : bool
        Whether to respace with Cartesian or angular spacing.
    eps : float
        The small adjustment subtracted from maxsep when respacing bins so that
        bins are actually a little closer together. This prevents computer
        precision issues from making bins too far apart.

    Returns
    -------
    (N,) ndarray
        Better-spaced bins. Bins are modified in-place.
    """
    # Initial state
    diffs = arccos(cos(diff(bins[:-1]))) if onsky else diff(bins[:-1])
    (seps,) = nonzero(diffs > maxsep)

    i = 0  # cap at 10k iterations
    while any(seps) and i < 10_000:

        # Move the bins by the separation
        bins[seps + 1] = bins[seps] + maxsep * (1 - eps)

        diffs = arccos(cos(diff(bins[:-1]))) if onsky else diff(bins[:-1])
        (seps,) = nonzero(diffs > maxsep)

        i += 1

    return bins


def _respace_bins(bins: NDT, maxsep: ndarray, onsky: bool, eps: float | np.floating) -> NDT:
    """Respace bins to have a maximum separation.

    Parameters
    ----------
    bins : (N,) ndarray
        The bins, which will be respaced.
    maxsep : scalar ndarray
        Maximum separation between bins
    onsky : bool
        Whether to respace with Cartesian or angular spacing.
    eps : float
        The small adjustment subtracted from maxsep when respacing bins so that
        bins are actually a little closer together. This prevents computer
        precision issues from making bins too far apart.

    Returns
    -------
    bins : (N,) ndarray
        Better-spaced bins. Bins are modified in-place.
    """
    # Check the separations
    diffs = arccos(cos(diff(bins))) if onsky else diff(bins)
    (seps,) = nonzero(diffs > maxsep)

    i = 0  # cap at 10k iterations
    while any(seps) and i < 50:

        # Adjust from the left, then adjust from the right
        bins = _respace_bins_from_left(bins, maxsep=maxsep, onsky=onsky, eps=eps)
        bins[::-1] = -_respace_bins_from_left(-bins[::-1], maxsep=maxsep, onsky=onsky, eps=eps)

        # Check the separations
        diffs = arccos(cos(diff(bins))) if onsky else diff(bins)
        (seps,) = nonzero(diffs > maxsep)

        i += 1

    return bins


def _decay_function(learning_rate: float, iteration: int, max_iter: float) -> float:
    """Decay function of the learning process.

    Parameters
    ----------
    learning_rate : float
        current learning rate.
    iteration : int
        current iteration.
    max_iter : int
        maximum number of iterations for the training.

    Returns
    -------
    float
    """
    return learning_rate / (1.0 + (2.0 * iteration / max_iter))


def _get_info_for_projection(
    data: ndarray,
    prototypes: ndarray,
) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    """Get orthogonal distance matrix for each point to each segment & node.

    Parameters
    ----------
    data : (N, D) ndarray
        The data. Rows are points, columns are features.
    prototypes : (P, D) ndarray


    Returns
    -------
    (N, PSN) ndarray
        Where P is the number of prototypes and connecting segments.
    """
    N, nF = data.shape
    nL = len(prototypes)

    # vector from one point to next  (nL-1, nF)
    p1 = prototypes[:-1, :]
    p2 = prototypes[1:, :]
    # vector from one point to next  (nL-1, nF)
    viip1 = np.subtract(p2, p1)
    # square distance from one point to next  (nL-1, nF)
    liip1 = np.sum(np.square(viip1), axis=1)

    # data - point_i  (N, nL-1, nF)
    dmi = np.subtract(data[:, None, :], p1[None, :, :])

    # The line extending the segment is parameterized as p1 + t (p2 - p1).
    # The projection falls where t = [(data-p1) . (p2-p1)] / |p2-p1|^2
    # tM is the matrix of "t"'s.
    tM = np.sum((dmi * viip1[None, :, :]), axis=-1) / liip1  # (N, nL-1)

    projected_points = p1[None, :, :] + tM[:, :, None] * viip1[None, :, :]

    # add in the nodes and find all the distances
    # the correct "place" to put the data point is within a
    # projection, unless it outside (by an endpoint)
    # or inside, but on the convex side of a segment junction
    all_points = np.empty((N, 2 * nL - 1, nF), dtype=float)
    all_points[:, 1::2, :] = projected_points
    all_points[:, 0::2, :] = prototypes
    distances = norm(np.subtract(data[:, None, :], all_points), axis=-1)
    # TODO! better on-sky treatment. This is a small-angle / flat-sky
    # approximation.

    # Detect whether it is in the segment. Nodes are considered in the segment. The end segments are allowed to extend.
    not_in_projection = np.zeros(all_points.shape[:-1], dtype=bool)
    not_in_projection[:, 1 + 2 : -2 : 2] = np.logical_or(tM[:, 1:-1] <= 0, 1 <= tM[:, 1:-1])
    not_in_projection[:, 1] = 1 <= tM[:, 1]  # end segs are 1/2 open
    not_in_projection[:, -2] = tM[:, -1] <= 0

    # make distances for not-in-segment infinity
    distances[not_in_projection] = np.inf

    return viip1, tM, all_points, distances


def _order_data_along_som_projection(
    data: ndarray, /, *, lattice_p2p_distance: ndarray, segment_projection: ndarray, distances: ndarray
) -> ndarray:
    r"""Order data along its projection onto 1D lattice.

    The curve is approximated by the linear segments connecting prototypes.

    Parameters
    ----------
    data : (N, D) ndarray[float]
        The data to order along the SOM projection
    lattice_p2p_distance : (P-1, D), ndarray
        Point-to-point distance between SOM prototypes.
    segment_projection : (N, P-1) ndarray[float]
        Each segment connecting protypes is parameterized as :math:`l_i(t_i) =
        p_i + t (p_{i+1} - p_{i})`. The projection of a point x falls where
        :math:`t_i = ((x-p_i) \cdot (p_{i+1}-p_i)] / |p_{i+1}-p_i|^2`.
        ``segment_projection`` is the matrix of these `t_i` for each point in ``data``.
    distances : (N, 2P-1, D) ndarray[float]
        Distance matrix of each point in ``data`` to every segment and node (prototype)

    Returns
    -------
    (N,) ndarray[int]
        The ordering of ``data``.
    """
    nlattice: int = len(lattice_p2p_distance) + 1

    # find the index of the best distance (including nodes)
    ind_best_distance = np.argmin(distances, axis=1)

    ordering = np.zeros(len(data), dtype=int) - 1

    counter = 0  # count through edge/node groups
    for i in np.unique(ind_best_distance):
        # for i in (1, ):
        # get the data rows for which the best distance is the i'th node/segment
        rowi = np.where(ind_best_distance == i)[0]
        numrows = len(rowi)

        # move edge points to corresponding segment
        if i == 0:
            i = 1
        elif i == 2 * (nlattice - 1):
            i = nlattice - 1

        # odds (remainder 1) are segments
        if bool(i % 2):
            ts = segment_projection[rowi, i // 2]
            rowsorter = np.argsort(ts)

        # evens are by nodes
        else:  # TODO! how many dimensions does this consider?
            phi1 = np.arctan2(*lattice_p2p_distance[i // 2 - 1, :2])
            phim2 = np.arctan2(*-lattice_p2p_distance[i // 2, :2])
            phi = np.arctan2(*data[rowi, :2].T)

            # detect if in branch cut territory
            if (np.pi / 2 <= phi1) & (-np.pi <= phim2) & (phim2 <= -np.pi / 2):
                phi1 -= 2 * np.pi
                phi -= 2 * np.pi

            rowsorter = np.argsort(phi) if phim2 < phi1 else np.argsort(-phi)

        slc = slice(counter, counter + numrows)
        ordering[slc] = rowi[rowsorter]

        counter += numrows

    ordering = np.array(ordering, dtype=int)

    return ordering


def project_data_on_som(prototypes: ndarray, data: ndarray) -> tuple[ndarray, ndarray]:
    """Project data onto a trained SOM.

    Parameters
    ----------
    prototypes : ndarray
        The SOM's prototypes.
    data : (N,) ndarray
        The data to project onto the SOM.

    Returns
    -------
    projected : (N,) ndarray
        The data projected onto the SOM. Not ordered.
    ordering : (N,) ndarray[int]
        The ordering of ``projected`` to ``data``.
    """
    lattice_p2p_distance, segment_projection, all_points, distances = _get_info_for_projection(data, prototypes)

    # projdata = _get_projected_point(som, arr, all_points, distances)
    ind_best_distance = np.argmin(distances, axis=1)
    projpnts = all_points[np.arange(len(distances)), ind_best_distance, :]

    ordering = _order_data_along_som_projection(
        data, lattice_p2p_distance=lattice_p2p_distance, segment_projection=segment_projection, distances=distances
    )
    return projpnts, ordering


# ===================================================================


def wrap_at(q: np.ndarray, /, wrap_angle: float) -> np.ndarray:
    """Wrap at value in radians.

    Function adapted from Astropy.

    Parameters
    ----------
    q : ndarray
        Units of radians. Must be subscriptable.
    wrap_angle : float
        Units of radians.
    """
    # Convert the wrap angle and 360 degrees to the native unit of
    # this Angle, then do all the math on raw Numpy arrays rather
    # than Quantity objects for speed.
    a360 = 2 * pi
    wrap_angle_floor = wrap_angle - a360
    # Do the wrapping, but only if any angles need to be wrapped.
    #
    # This invalid catch block is needed both for the floor division
    # and for the comparisons later on (latter not really needed
    # any more for >= 1.19 (NUMPY_LT_1_19), but former is).
    with np.errstate(invalid="ignore"):
        wraps = (q - wrap_angle_floor) // a360
        np.nan_to_num(wraps, copy=False)
        if np.any(wraps != 0):
            q -= wraps * a360
            # Rounding errors can cause problems.
            q[q >= wrap_angle] -= a360
            q[q < wrap_angle_floor] += a360
    return q
