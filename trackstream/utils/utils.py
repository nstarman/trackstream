# -*- coding: utf-8 -*-

"""Utilities for :mod:`~trackstream.utils`."""


__all__ = [
    "intermix_arrays",
]


##############################################################################
# IMPORTS

# BUILT-IN
import typing as T

# THIRD PARTY
import numpy as np

##############################################################################
# PARAMETERS


##############################################################################
# CODE
##############################################################################

def make_shuffler(
    length: int, rng=None
) -> T.Tuple[T.Sequence[int], T.Sequence[int]]:
    """
    Shuffle and Unshuffle arrays.

    Parameters
    ----------
    length : int
        Array length for which to construct (un)shuffle arrays.
    rng : :class:`~numpy.random.Generator` instance, optional
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
        try:
            rng = np.random.default_rng()
        except AttributeError:
            rng = np.random

    # start with index array
    shuffler = np.arange(length)
    # now shuffle array (in-place)
    rng.shuffle(shuffler)

    # and construct the unshuffler
    undo = shuffler.argsort()

    return shuffler, undo


# /def

##############################################################################
# END
