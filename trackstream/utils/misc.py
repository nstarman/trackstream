# -*- coding: utf-8 -*-

"""Utilities for :mod:`~trackstream.utils`."""


__all__ = ["intermix_arrays", "make_shuffler"]


##############################################################################
# IMPORTS

# STDLIB
from typing import Optional, Sequence, Tuple, Union

# THIRD PARTY
import numpy as np
from numpy.random import Generator

##############################################################################
# CODE
##############################################################################


def intermix_arrays(*arrs: Union[Sequence, np.ndarray], axis: int = -1) -> np.ndarray:
    """Intermix arrays.

    Parameters
    ----------
    *arrs : Sequence
    axis : int, optional

    Return
    ------
    arr : ndarray

    Examples
    --------
    Mix single scalar array (does nothing)

        >>> x = np.arange(5)
        >>> intermix_arrays(x)
        array([0, 1, 2, 3, 4])

    Mix two scalar arrays

        >>> y = np.arange(5, 10)
        >>> intermix_arrays(x, y)
        array([0, 5, 1, 6, 2, 7, 3, 8, 4, 9])

    Mix multiple scalar arrays

        >>> z = np.arange(10, 15)
        >>> intermix_arrays(x, y, z)
        array([ 0,  5, 10,  1,  6, 11,  2,  7, 12,  3,  8, 13,  4,  9, 14])

    Mix single ND array

        >>> xx = np.c_[x, y]
        >>> intermix_arrays(xx)
        array([[0, 1, 2, 3, 4],
               [5, 6, 7, 8, 9]])

    Mix two ND arrays

        >>> yy = np.c_[z, np.arange(15, 20)]
        >>> intermix_arrays(xx, yy)
        array([[ 0, 10,  1, 11,  2, 12,  3, 13,  4, 14],
               [ 5, 15,  6, 16,  7, 17,  8, 18,  9, 19]])
    """
    shape = list(np.asanyarray(arrs[0]).shape[::-1])
    shape[axis] *= len(arrs)

    return np.asanyarray(arrs).T.flatten().reshape(shape)


def make_shuffler(
    length: int,
    rng: Optional[Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Shuffle and un-shuffle arrays.

    Parameters
    ----------
    length : int
        Array length for which to construct (un)shuffle arrays.
    rng : `~numpy.random.Generator`, optional
        random number generator.

    Returns
    -------
    shuffler : `~numpy.ndarray`
        index array that shuffles any array of size `length` along
        a specified axis
    undo : `~numpy.ndarray`
        index array that undoes above, if applied identically.
    """
    if rng is None:
        rng = np.random.default_rng()

    shuffler = np.arange(length)  # start with index array
    rng.shuffle(shuffler)  # shuffle array in-place

    undo = shuffler.argsort()  # and construct the un-shuffler

    return shuffler, undo
