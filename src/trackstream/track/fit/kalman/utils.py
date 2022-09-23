"""Coordinates Utilities."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import warnings
from typing import TYPE_CHECKING, Any

# THIRD PARTY
import numpy as np
from numpy import asanyarray

# LOCAL
from trackstream.utils.coord_utils import f2q
from trackstream.utils.unit_utils import merge_units

if TYPE_CHECKING:
    # THIRD PARTY
    from astropy.units import Quantity
    from numpy.typing import ArrayLike, NDArray

    # LOCAL
    from trackstream.stream.core import StreamArm
    from trackstream.track.fit.kalman.core import FirstOrderNewtonianKalmanFilter

__all__: list[str] = []


##############################################################################
# CODE
##############################################################################


def intermix_arrays(*arrs: ArrayLike, axis: int = -1) -> NDArray[Any]:
    """Intermix arrays.

    Parameters
    ----------
    *arrs : (N,) array-like
        All arrays should be the same length.
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


def make_error(stream: StreamArm, kf: FirstOrderNewtonianKalmanFilter, default: float = 0) -> Quantity:
    """Get Error.

    Parameters
    ----------
    data : `~astropy.table.QTable`
        The data table from which to extract the error information.
    kf : FirstOrderNewtonianKalmanFilter
        The Kalman filter to use to understand what information should be gotten
        from ``data``.
    default : float, optional
        Default error value, by default 0.

    Returns
    -------
    (N,) Quantity
        A structured quantity with D fields labeled by ``kf.info.units``.
    """
    flat_units = merge_units(kf.info.units)
    errors = np.zeros(len(stream), dtype=[(n, float) for n in flat_units.field_names])

    # TODO! without transforming the data. See ``get_representation_names``.
    crds = stream.coords
    crds.representation_type = kf.info.representation_type
    crds.differential_type = kf.info.differential_type
    svs = f2q(crds, flatten=True)

    for rn, fn in zip(kf.info.components(True), svs.dtype.names):
        unit = flat_units[rn]

        if (fne := f"{fn}_err") in stream.data.columns:
            r = stream.data[fne].to_value(unit)
        elif (rne := f"{rn}_err") in stream.data.columns:
            r = stream.data[fne].to_value(unit)
        else:
            msg = f"{fne} and {rne} are not in the data; setting to the default."
            warnings.warn(msg)
            r = default

        errors[rn] = r**2

    return errors << flat_units
