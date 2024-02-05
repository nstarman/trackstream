"""Coordinates Utilities."""

from __future__ import annotations

from collections.abc import (
    Callable,
    Iterator,
    KeysView,
    Mapping,
    MutableMapping,
    ValuesView,
)
from typing import TYPE_CHECKING, Any, TypeVar, cast

import astropy.units as u
from numpy import any, arctan2, ndarray, pi, sqrt, vectorize
from scipy.linalg import svd

if TYPE_CHECKING:
    from astropy.units import Quantity

__all__: list[str] = []

##############################################################################
# PARAMETERS

V = TypeVar("V")
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
    """Compute the covariance ellipse.

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
        msg = "width must be greater than height"
        raise ValueError(msg)

    return orientation, wh


##############################################################################


class PhysicalTypeKeyMapping(Mapping[u.PhysicalType, V]):
    """Mapping with PhysicalType keys."""

    def __init__(self, mapping: Mapping[u.PhysicalType, V], /) -> None:
        self._mapping = mapping

    @staticmethod
    def _get_key(key: str | u.PhysicalType) -> u.PhysicalType:
        return key if isinstance(key, u.PhysicalType) else cast("u.PhysicalType", u.get_physical_type(key))

    def __getitem__(self, key: str | u.PhysicalType, /) -> V:
        return self._mapping[self._get_key(key)]

    def __iter__(self) -> Iterator[u.PhysicalType]:
        return iter(self._mapping)

    def __len__(self) -> int:
        return len(self._mapping)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._mapping!r})"

    def keys(self) -> KeysView[u.PhysicalType]:
        """Return the keys."""
        return self._mapping.keys()

    def values(self) -> ValuesView[V]:
        """Return the values."""
        return self._mapping.values()


class PhysicalTypeKeyMutableMapping(PhysicalTypeKeyMapping[V], MutableMapping[u.PhysicalType, V]):
    """Mutable mapping with PhysicalType keys."""

    def __init__(self, mapping: MutableMapping[u.PhysicalType, V], /) -> None:
        self._mapping: MutableMapping[u.PhysicalType, V] = mapping

    def __setitem__(self, key: str | u.PhysicalType, value: V, /) -> None:
        self._mapping[self._get_key(key)] = value

    def __delitem__(self, k: str | u.PhysicalType) -> None:
        del self._mapping[self._get_key(k)]
