# -*- coding: utf-8 -*-

"""Preprocessing.

TODO find slowest lines
https://marcobonzanini.com/2015/01/05/my-python-code-is-slow-tips-for-profiling/

"""


__all__ = [
    # functions
    "apply_SOM",
    "apply_SOM_repeat",
    "make_transition_matrix",
    "draw_ordering",
    "draw_most_probable_ordering",
    "preprocess",
]


##############################################################################
# IMPORTS

# BUILT-IN

# BUILT-IN
import typing as T
import warnings
from collections import namedtuple

# THIRD PARTY
import astropy.coordinates as coord
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from tqdm import tqdm

from ..conf import conf

# PROJECT-SPECIFIC


if conf.use_minisom:
    try:
        # THIRD PARTY
        from minisom import MiniSom as SOM
    except ImportError:
        warnings.warn("Can't find MiniSOM, falling back to built-in SOM.")
        from .som import SelfOrganizingMap as SOM
else:
    from .som import SelfOrganizingMap as SOM

from .som import reorder_visits
from .utils import CoordinateType, DataType, find_closest_point, set_starting_point

##############################################################################
# PARAMETERS

warnings.filterwarnings(
    "ignore", message="Warning: sigma is too high for the dimension of the map.",
)


preprocessed = namedtuple(
    "preprocessed", ("data", "trmat", "visit_orders", "start_point")
)


##############################################################################
# CODE
##############################################################################


def apply_SOM(
    data: CoordinateType,
    *,
    learning_rate: float = 2.0,
    sigma: float = 20.0,
    iterations: int = 10000,
    random_seed: T.Optional[int] = None,
    reorder: T.Optional[int] = None,
    plot: bool = False,
    return_som: bool = False,
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
        data, (coord.SkyCoord, coord.BaseCoordinateFrame, coord.BaseRepresentation),
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

    som = SOM(
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

        ax.plot(data[order][:, 0], data[order][:, 1], c="gray")
        # ax.scatter(data[order[0], 0], data[order[0], 1], c="g")
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

    if return_som:
        return order, som

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
        data, (coord.SkyCoord, coord.BaseCoordinateFrame, coord.BaseRepresentation),
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
            data[:, 0], data[:, 1], c="k", vmax=len(data), cmap="plasma", label="data",
        )

        for order in orders:
            ax.plot(data[order][:, 0], data[order][:, 1], c="gray", alpha=0.5)
            ax.scatter(data[order][0, 0], data[order][-1, 1], c="g")

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


def draw_ordering(trmat, num: int = 1, rng: T.Optional[np.random.Generator] = None):
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
    # flipped to get half-open bound (0, 1]
    alpha = rng.uniform(low=1.0, high=0.0, size=shape)
    orders = np.full(shape, -1, dtype=int)

    # iterating through rows of trmat, selecting the transition pair.
    for i, (inds, probs) in enumerate(zip(trmat.rows, trmat.data)):
        if probs[0] == 1.0:  # easy exit
            orders[i] = inds[0]
            continue
        # select among steps by probs
        sel = np.cumsum(probs) >= alpha[i][:, None]  # shape st rows=num
        # get first index for which sel is True
        orders[i] = [inds[i] for i in np.argmax(sel, axis=1)]  # for rows

    # TODO check no -1s left from the np.full
    if np.any(orders == -1):
        raise Exception("Something went wrong in `draw_ordering`")

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
    data: CoordinateType,
    start_point: T.Sequence,
    N_repeats: T.Union[int, T.Sequence] = 10,
    *,
    iterations: int = int(3e3),
    learning_rate: float = 1.5,
    sigma: float = 15,
    plot: bool = True,
    _tqdm: bool = True,
):
    """Preprocess.

    Parameters
    ----------
    data : CoordinateType
    start_point : Sequence
        The coordinates of the starting point.
    N_repeats : int or Sequence, optional
        The number of times to do the SOM, setting the random seeds.
        If sequence, interpreted as the random seeds, where number of
        repeats is length of sequence.

    iterations : int, optional, keyword only
        Number of iterations for each SOM.
    learning_rate : float, optional, keyword only
        SOM learning rate.
    sigma : float, optional, keyword only
        SOM sigma.
    plot : bool, optional, keyword only
        Whether to plot the preprocessing results.

    Returns
    -------
    visit_orders

    Other Parameters
    ----------------
    _tqdm : bool, optional, keyword only
        Whether to use tqdm progress bar

    """
    # ---------------
    # starting point
    start_point = np.asanyarray(start_point)  # lets units through

    start_ind: int
    start_point, start_ind = find_closest_point(data, start_point)

    data = set_starting_point(data, start_ind)
    start_ind: int = 0

    # ---------------
    # SOM

    if isinstance(N_repeats, int):
        random_seeds = np.arange(0, N_repeats, 1)
    else:
        random_seeds = N_repeats

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

    # ---------------
    # Transition Matrix

    trmat = make_transition_matrix(visit_orders)

    # ---------------

    return preprocessed(
        data=data, trmat=trmat, visit_orders=visit_orders, start_point=start_point,
    )


# /def


##############################################################################
# END
