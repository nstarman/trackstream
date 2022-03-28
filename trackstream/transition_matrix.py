# -*- coding: utf-8 -*-

"""Transition matrix from SOM."""

__all__ = [
    "make_transition_matrix",
    "draw_ordering",
    "draw_most_probable_ordering",
]

##############################################################################
# IMPORTS

# STDLIB
from typing import Optional, Sequence

# THIRD PARTY
import numpy as np
from scipy import sparse

##############################################################################
# CODE
##############################################################################


def make_transition_matrix(orders: Sequence[Sequence]) -> sparse.lil_matrix:
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
    trmat = sparse.lil_matrix((nelt, nelt), dtype=float)  # empty, sparse

    # fill in transition pairs, counting number of occurrences.
    for i in range(nelt):  # TODO vectorize
        visited, counts = np.unique(orders[:, i], return_counts=True)
        trmat[i, visited] += counts

    # convert from count to probability by dividing by number of orderings.
    trmat /= orders.shape[0]

    return trmat


def draw_ordering(
    trmat,
    num: int = 1,
    rng: Optional[np.random.Generator] = None,
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


##############################################################################
# END
