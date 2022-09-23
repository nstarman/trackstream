"""Coordinates Utilities."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from typing import TYPE_CHECKING, Any, Callable, TypeVar

# THIRD PARTY
import astropy.units as u
from numpy import any, arctan2, ndarray, pi, sqrt, vectorize
from scipy.linalg import svd

if TYPE_CHECKING:
    # THIRD PARTY
    from astropy.units import Quantity

__all__: list[str] = []

##############################################################################
# PARAMETERS

R = TypeVar("R")  # return variable

_PI_2 = pi / 2

##############################################################################
# CODE
##############################################################################


def is_structured(x: Any, /) -> bool:
    """Return whether ``x`` is a structured array."""
    return getattr(getattr(x, "dtype", None), "names", None) is not None


# -------------------------------------------------------------------


svd_vec: Callable = vectorize(
    svd,
    otypes=[float, float, float],
    excluded={"full_matrices", "compute_uv", "overwrite_a", "check_finite", "lapack_driver"},
    signature=("(m,n)->(m,m),(m),(n,n)"),
)


def covariance_ellipse(P: ndarray, *, nstd: int | ndarray = 1) -> tuple[Quantity, ndarray]:
    """
    Returns a tuple defining the ellipse representing the 2 dimensional
    covariance matrix P.

    Parameters
    ----------
    P : (N?, M, M) ndarray
       covariance matrix(ces).
    nstd : int or (N?,) ndarray, optional keyword-only
       Number of standard deviations. Default is 1.

    Returns
    -------
    orientation : (N?,) Quantity['angle']
        The angle.
    wh : (N?, 2) ndarray
        width and height radius.

    Notes
    -----
    Modified from :mod:`filterpy`
    """
    U, s, _ = svd_vec(P)  # requires (d1, d2) matrix

    orientation = arctan2(U[..., 1, 0], U[..., 0, 0]) << u.rad
    wh = nstd * sqrt(s[..., :2])

    if any(wh[..., 1] > wh[..., 0]):
        raise ValueError("width must be greater than height")

    return orientation, wh
