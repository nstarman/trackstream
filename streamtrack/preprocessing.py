# -*- coding: utf-8 -*-

"""Preprocessing.

TODO find slowest lines
https://marcobonzanini.com/2015/01/05/my-python-code-is-slow-tips-for-profiling/

"""


__all__ = [
    # functions
    "find_starting_point",
    "set_starting_point",
    "reorder_visits",
    "apply_SOM",
    "apply_SOM_repeat",
    "make_transition_matrix",
    "draw_ordering",
]


##############################################################################
# IMPORTS

# BUILT-IN

import typing as T


# THIRD PARTY

import astropy.coordinates as coord

from minisom import MiniSom

import matplotlib.pyplot as plt

import numpy as np

from scipy import sparse

from sklearn.neighbors import KDTree

from tqdm import tqdm

from utilipy.utils.typing import CoordinateType, CoordinateRepresentationType

import warnings

# PROJECT-SPECIFIC


##############################################################################
# PARAMETERS

warnings.filterwarnings(
    "ignore",
    message="Warning: sigma is too high for the dimension of the map.",
)


DataType = T.Union[T.Sequence, CoordinateType, CoordinateRepresentationType]


##############################################################################
# CODE
##############################################################################


def _find_starting_point(
    data: T.Sequence, near_point: T.Sequence, return_kdtree: bool = False,
):
    """Find starting point.

    Uses a k-d tree to query for the point in `data` closes to `near_point`.

    Parameters
    ----------
    data : Sequence
        Shape (# measurements, # features)
        Must be in Cartesian coordinates for the KDTree distance function.
    near_point : Sequence
        Shape (1, # features)
        If passing an array, can reshape with ``.reshape(1, -1)``
    return_ind : bool
        Whether to return index.

    Returns
    -------
    (start_point, start_ind) : if `return_kdtree` is False
    (start_point, start_ind, tree) : if `return_ind` = `return_kdtree` = True

    start_point : Sequence
        Shape (# features, ). Point in `data` nearest `near_point` in KDTree
    start_ind : int
        Index into `data` for the `start_point`
        If `return_ind` == True
    tree : `sklearn.neighbors.KDTree` instance

    """
    # build tree
    tree = KDTree(data)
    # find close points and their indices in data
    dists, inds = tree.query(near_point, k=4, return_distance=True)

    # get closest point & index
    start_ind = inds[0][np.argmin(dists[0])]
    start_point = data[start_ind]

    # ----------
    # Return

    if return_kdtree:
        return start_point, start_ind, tree

    return start_point, start_ind


# /def


def find_starting_point(
    data: DataType, near_point: T.Sequence, return_kdtree: bool = False,
):
    """Find starting point.

    .. |Rep| replace:: :class:`~astropy.coordinates.BaseRepresentation`
    .. |Coord| replace:: :class:`~astropy.coordinates.BaseCoordinateFrame`

    Parameters
    ----------
    data : |Rep| or |Coord| instance or Sequence
        Shape (# measurements, # features).
        Must be transformable to Cartesian coordinates.
    near_point : Sequence
        Shape (1, # features)
        If passing an array, can reshape with ``.reshape(1, -1)``
    return_ind : bool
        Whether to return index.

    Returns
    -------
    (start_point, start_ind) : if `return_kdtree` is False
    (start_point, start_ind, tree) : if `return_ind` = `return_kdtree` = True

    start_point : Sequence
        Shape (# features, ). Point in `data` nearest `near_point` in KDTree
    start_ind : int
        Index into `data` for the `start_point`
        If `return_ind` == True
    tree : `sklearn.neighbors.KDTree` instance

    """
    # ----------
    # Conversion

    if isinstance(
        data,
        (coord.SkyCoord, coord.BaseCoordinateFrame, coord.BaseRepresentation),
    ):
        rep = data.represent_as(coord.CartesianRepresentation)
        data = rep._values.view("f8").reshape(-1, len(rep.components))

    # ----------
    # Return

    return _find_starting_point(
        data=data, near_point=near_point, return_kdtree=return_kdtree
    )


# /def


# -------------------------------------------------------------------


def set_starting_point(data: DataType, start_ind: int):
    """Reorder data to set starting index at row 0.

    Parameters
    ----------
    data
    start_ind

    Returns
    -------
    `data`
        re-ordered

    """
    # index order array
    order = list(range(len(data)))
    del order[start_ind]
    order = np.array([start_ind, *order])

    return data[order]  # return reordered data


# /def


# -------------------------------------------------------------------


def reorder_visits(
    data: CoordinateType, visit_order: T.Sequence, start_ind: int,
):
    """Reorder the points from the SOM.

    The SOM does not always keep the starting point at the beginning
    nor even "direction" of the indices. This function can flip the
    index ordering and rotate the indices such that the starting point
    stays at the beginning.

    The rotation is done by snipping and restitching the array at
    `start_ind`. The "direction" is determined by the distance between
    data points at visit_order[start_ind-1] and visit_order[start_ind+1].

    Parameters
    ----------
    data : CoordinateType
        The data.
    visit_order: Sequence
        Index array ordering `data`. Will be flipped and / or rotated
        such that `start_ind` is the 0th element and the next element
        is the closest.
    start_ind : int
        The starting index

    Returns
    -------
    new_visit_order : Sequence
        reordering of `visit_order`

    """
    # index of start_ind in visit_order
    # this needs to be made the first index.
    i = list(visit_order).index(start_ind)

    # snipping and restitching
    # first taking care of the edge cases
    if i == 0:  # AOK
        pass
    elif i == len(visit_order) - 1:  # just reverse.
        i = 0
        visit_order = visit_order[::-1]
    else:  # need to figure out direction before restitching
        back = (data[visit_order[i]] - data[visit_order[i - 1]]).norm()
        forw = (data[visit_order[i + 1]] - data[visit_order[i]]).norm()

        if back < forw:  # things got reversed...
            visit_order = visit_order[::-1]  # flip visit_order
            i = list(visit_order).index(start_ind)  # re-find index

    # do the stitch
    new_visit_order = np.concatenate((visit_order[i:], visit_order[:i]))

    return new_visit_order


# /def


# -------------------------------------------------------------------


def apply_SOM(
    data: CoordinateType,
    *,
    learning_rate: float = 2.0,
    sigma: float = 20.0,
    iterations: int = 10000,
    random_seed: T.Optional[int] = None,
    reorder: T.Optional[int] = None,
    plot: bool = False,
) -> T.Sequence[int]:
    """Apply Self Ordered Mapping to the data.

    .. |Rep| replace:: :class:`~astropy.coordinates.BaseRepresentation`
    .. |Coord| replace:: :class:`~astropy.coordinates.BaseCoordinateFrame`
    .. |SkyCoord| replace:: :class:`~astropy.coordinates.SkyCoord`

    Parameters
    ----------
    data : |Rep| or |Coord| or |SkyCoord|
        The data.
    learning_rate : float, optional, keyword only
        (at the iteration t we have
        learning_rate(t) = learning_rate / (1 + t/T)
        where T is #num_iteration/2)
    sigma : float, optional, keyword only
        Spread of the neighborhood function, needs to be adequate
        to the dimensions of the map.
        (at the iteration t we have sigma(t) = sigma / (1 + t/T)
        where T is #num_iteration/2)
    iterations : int, optional, keyword only
        number of times the SOM is trained.
    random_seed: int, optional, keyword only
        Random seed to use. (default=None).
    reorder : int, optional, keyword only
        If not None (the default), the starting index.

    Returns
    -------
    order : Sequence

    Other Parameters
    ----------------
    plot : bool, optional, keyword only
        Whether to plot the results

    Raises
    ------
    TypeError
        If `data` is not a |Rep| or |Coord| or |SkyCoord|
        and `reorder` is not None.

    """
    # ----------
    # Conversion

    if isinstance(
        data,
        (coord.SkyCoord, coord.BaseCoordinateFrame, coord.BaseRepresentation),
    ):
        rep = data.represent_as(coord.CartesianRepresentation)
        data = rep._values.view("f8").reshape(-1, len(rep.components))
    elif reorder is None:  # only need to enforce data-type if reorder not None
        pass
    else:
        raise TypeError

    # -------
    # SOM

    stream_dims = 1  # streams are 1D

    data_len = data.shape[0]  # length of data
    nfeature = data.shape[1]  # of features: (x, y, z) or (ra, dec), etc.

    som = MiniSom(
        x=stream_dims,
        y=nfeature * data_len,
        input_len=nfeature,
        sigma=sigma,
        learning_rate=learning_rate,
        # decay_function=None,
        neighborhood_function="gaussian",
        topology="rectangular",
        activation_distance="euclidean",
        random_seed=random_seed,
    )
    som.pca_weights_init(data)

    # train the data, preserving order
    som.train(data, iterations, verbose=False, random_order=False)

    # get the ordering by "vote" of the Prototypes
    visit_order = np.argsort([som.winner(p)[1] for p in data])

    # ---------------
    # Reorder

    if reorder is not None:
        order = reorder_visits(rep, visit_order, start_ind=reorder)

    else:  # don't reorder
        order = visit_order

    # ---------------
    # Plot

    if plot:

        fig, ax = plt.subplots(figsize=(10, 9))

        pts = ax.scatter(
            data[order, 0],
            data[order, 1],
            c=np.arange(0, len(data)),
            vmax=len(data),
            cmap="plasma",
            label="data",
        )
        # ax.scatter(data[:, 0], data[:, 1], c="k", s=1, label="data")

        ax.plot(data[order][:, 0], data[order][:, 1], c="gray")
        ax.set_title(
            "iterations: {i};\nerror: {e:.3f}".format(
                i=iterations, e=som.quantization_error(data)
            )
        )

        cbar = plt.colorbar(pts, ax=ax)
        cbar.ax.set_ylabel("SOM ordering")

        fig.legend(loc="upper left")
        fig.tight_layout()
        plt.show()

    # /if

    # ---------------

    return order


# /def


# -------------------------------------------------------------------


def apply_SOM_repeat(
    data: DataType,
    random_seeds: T.Optional[T.Sequence[int]],
    *,
    dims: int = 1,
    learning_rate: float = 0.8,
    sigma: float = 4.0,
    iterations: int = 10000,
    reorder: T.Optional[T.Sequence] = True,
    plot: bool = False,
    _tqdm: bool = True,
) -> T.Sequence[int]:
    """SOM Preprocess.

    Parameters
    ----------
    data : BaseCoordinateFrame instance
    random_seeds: Sequence
        Random seeds to use.

    dims : int, optional, keyword only
    learning_rate : float, optional, keyword only
    sigma : float, optional, keyword only
        Spread of the neighborhood function, needs to be adequate
        to the dimensions of the map.
        (at the iteration t we have sigma(t) = sigma / (1 + t/T)
        where T is #num_iteration/2)
    neighborhood_function : str, optional, keyword only
    iterations : int, optional, keyword only

    reorder : Sequence, optional, keyword only
        If not None, the starting index.
    plot : bool, optional, keyword only

    Returns
    -------
    orders : Sequence
        Shape (len(`random_seeds`), len(`data`))

    """
    if isinstance(
        data,
        (coord.SkyCoord, coord.BaseCoordinateFrame, coord.BaseRepresentation),
    ):
        rep = data.represent_as(coord.CartesianRepresentation)
        data = rep._values.view("f8").reshape(-1, len(rep.components))
    else:
        raise TypeError

    # -------

    nrows = len(random_seeds)
    orders = np.empty((nrows, len(data)), dtype=int)

    iterator = tqdm(random_seeds) if _tqdm else random_seeds
    for i, seed in enumerate(iterator):
        visit_order = apply_SOM(
            data,
            learning_rate=learning_rate,
            sigma=sigma,
            iterations=iterations,
            random_seed=seed,
            reorder=None,
            plot=False,
        )

        if reorder is not None:
            order = reorder_visits(rep, visit_order, start_ind=reorder)

        else:
            order = visit_order

        orders[i] = order

    # /for

    # ---------------

    if plot:

        fig, ax = plt.subplots(figsize=(10, 9))

        ax.scatter(
            data[:, 0],
            data[:, 1],
            c="k",
            vmax=len(data),
            cmap="plasma",
            label="data",
        )

        for order in orders:
            ax.plot(data[order][:, 0], data[order][:, 1], c="gray", alpha=0.5)

        fig.legend(loc="upper left")
        fig.tight_layout()
        plt.show()

    # /if

    # ---------------

    return orders


# /def


# -------------------------------------------------------------------


def make_transition_matrix(orders: T.Sequence[T.Sequence]):
    """Make Transition Matrix from SOM-derived orders.

    The SOM-derived orders can vary with the random seed. To account
    for the non-determinism we construct a transition matrix that
    gives the probability of going from index *i* to *j*. For this,
    we need many SOM-derived orders.

    Parameters
    ----------
    orders: list of lists
        Shape (N, len(data))

    Returns
    -------
    trmat : `~scipy.sparse.lil_matrix`
        The sparse transition matrix. All rows and columns sum to 1.
        Row index is for *i*, column index for *j*.
        Shape (len(data), len(data)).

    See Also
    --------
    `~draw_ordering`

    """
    nelt = orders.shape[1]  # number of elements in matrix

    # empty and sparse transition matrix
    trmat = sparse.lil_matrix((nelt, nelt), dtype=float)

    # fill in transition pairs, counting number of occurrences.
    for i in range(nelt):  # TODO vectorize
        visited, counts = np.unique(orders[:, i], return_counts=True)
        trmat[i, visited] += counts

    # convert from count to probability by dividing by number of orderings.
    trmat /= orders.shape[0]

    return trmat


# /def


def draw_ordering(
    trmat, num: int = 1, rng: T.Optional[np.random.Generator] = None
):
    """Draw ordering(s) from transition matrix.

    Parameters
    ----------
    trmat : ndarray
        Transition matrix. square.
    num : int, optional
        number of orderings to draw
    rng : Generator instance, optional
        Random number generator.

    Returns
    -------
    orders : `~numpy.ndarray`
        Shape (`num`, len(`trmat`)).

    Notes
    -----
    The orderings are drawn by iterating through each row *i* of `trmat`,
    which encodes the index in the data. The columns *j* of each row
    encode the next index in the ordering, with some probability
    :math:`p_{i,j}`. The probabilities are accumulated (summing to 1)
    and a random number is uniformly generated to select the *j* index.

    """
    rng = rng if rng is not None else np.random.default_rng()
    size = trmat.shape[0]
    shape = (size, num)

    # The selection function for which index *j* to select in each row *i*
    alpha = rng.uniform(
        low=1.0, high=0.0, size=shape
    )  # flipped to get half-open bound (0, 1]
    orders = np.full(shape, -1, dtype=int)

    # TODO vectorize this for loop
    # iterating through rows of trmat, selecting the transition pair.
    for i, (inds, probs) in enumerate(zip(trmat.rows, trmat.data)):
        if probs[0] == 1.0:
            orders[i] = inds[0]
            continue
        # select among steps by probs
        sel = np.cumsum(probs) >= alpha[i][:, None]  # shape st rows=num
        # get first index for which sel is True
        orders[i] = [inds[i] for i in np.argmax(sel, axis=1)]  # for rows

    # TODO check no -1s left from the np.full

    return orders.T


# /def


# -------------------------------------------------------------------


def draw_most_probable_ordering(trmat):
    """Draw most probably ordering from transition matrix.

    Parameters
    ----------
    trmat : ndarray
        Transition matrix. square.
    num : int, optional
        number of orderings to draw
    rng : Generator instance, optional
        Random number generator.

    Returns
    -------
    best_order : `~numpy.ndarray`
        Same length as `trmat`.

    Notes
    -----
    The orderings are drawn by iterating through each row *i* of `trmat`,
    which encodes the index in the data. The columns *j* of each row
    encode the next index in the ordering, with some probability
    :math:`p_{i,j}`. The probabilities are accumulated (summing to 1)
    and a random number is uniformly generated to select the *j* index.

    """
    best_order = np.full(trmat.shape[0], -1, dtype=int)

    # TODO vectorize this for loop
    # iterating through rows of trmat, selecting the transition pair.
    for i, (inds, probs) in enumerate(zip(trmat.rows, trmat.data)):
        # select among steps by most probable
        best_order[i] = inds[np.argmax(probs)]

    return best_order


# /def

# -------------------------------------------------------------------


def preprocess(
    data,
    start_point,
    *,
    iterations=int(3e3),
    learning_rate=1.5,
    sigma=15,
    plot=True,
    random_seeds=np.arange(0, 10, 1),

    _tqdm=True,
):
    """Preprocess.

    Parameters
    ----------
    data
    start_point
    iterations
    learning_rate
    sigma
    plot
    random_seeds

    Returns
    -------
    visit_orders

    Other Parameters
    ----------------
    _tqdm

    """
    start_ind: int
    start_point, start_ind, tree = find_starting_point(
        data, start_point, return_kdtree=True
    )

    data = set_starting_point(data, start_ind)
    start_ind: int = 0

    visit_orders = apply_SOM_repeat(
        data,
        random_seeds=random_seeds,
        iterations=iterations,
        learning_rate=learning_rate,
        sigma=sigma,
        plot=plot,
        reorder=start_ind,
        _tqdm=_tqdm,
    )

    trmat = make_transition_matrix(visit_orders)

    return data, trmat, visit_orders


# /def


##############################################################################
# END
