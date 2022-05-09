# -*- coding: utf-8 -*-

"""Utilities for :mod:`~trackstream.utils`."""


__all__ = ["intermix_arrays", "make_shuffler", "abstract_attribute", "is_structured"]


##############################################################################
# IMPORTS

# STDLIB
from typing import Any, Callable, Optional, Sequence, Tuple, TypeVar, Union, cast

# THIRD PARTY
import astropy.units as u
from astropy.units import Quantity
from numpy import arange, arctan2, asanyarray, ndarray, sqrt, vectorize
from numpy.random import Generator, default_rng
from scipy.linalg import svd

# LOCAL
from trackstream._type_hints import DummyAttribute

##############################################################################
# PARAMETERS

R = TypeVar("R")  # return variable


##############################################################################
# CODE
##############################################################################


def intermix_arrays(*arrs: Union[Sequence, ndarray], axis: int = -1) -> ndarray:
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
    shape = list(asanyarray(arrs[0]).shape[::-1])
    shape[axis] *= len(arrs)

    return asanyarray(arrs).T.flatten().reshape(shape)


def make_shuffler(
    length: int,
    rng: Optional[Generator] = None,
) -> Tuple[ndarray, ndarray]:
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
        rng = default_rng()

    shuffler = arange(length)  # start with index array
    rng.shuffle(shuffler)  # shuffle array in-place

    undo = shuffler.argsort()  # and construct the un-shuffler

    return shuffler, undo


def abstract_attribute(obj: Optional[Callable[[Any], R]] = None) -> R:
    # https://stackoverflow.com/a/50381071
    _obj = cast(Any, obj)  # prevent complaint about assigning attributes
    if obj is None:
        _obj = DummyAttribute()
    _obj.__is_abstract_attribute__ = True
    return cast(R, _obj)


def is_structured(x: Any, /) -> bool:
    """Return whether ``x`` is a structured array."""
    return getattr(getattr(x, "dtype", None), "names", None) is not None


svd_vec: Callable = vectorize(
    svd,
    otypes=[float, float, float],
    excluded={"full_matrices", "compute_uv", "overwrite_a", "check_finite", "lapack_driver"},
    signature=("(m,n)->(m,m),(m),(n,n)"),
)


def covariance_ellipse(P: ndarray, deviations: Union[int, ndarray] = 1) -> Tuple[Quantity, ndarray]:
    """
    FROM FILTERPY

    Returns a tuple defining the ellipse representing the 2 dimensional
    covariance matrix P.

    Parameters
    ----------
    P : (N?, M, M) ndarray
       covariance matrix(ces).

    deviations : int or (N?,) ndarray, optional
       Number of standard deviations. Default is 1.

    Returns
    -------
    angle_radians : (N?,) Quantity
    wh : (N?, 2) ndarray
        width and height radius
    """
    U, s, _ = svd_vec(P)  # requires (d1, d2) matrix

    orientation = arctan2(U[..., 1, 0], U[..., 0, 0]) * u.rad
    wh = deviations * sqrt(s[..., :2])

    if any(wh[..., 1] > wh[..., 0]):
        raise ValueError("width must be greater than height")

    return orientation, wh
