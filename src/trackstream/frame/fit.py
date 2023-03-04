"""Fit a Rotated reference frame."""


from __future__ import annotations

from collections.abc import Callable, Mapping
import functools
from typing import TYPE_CHECKING, Any, TypedDict, cast

import astropy.coordinates as coords
from astropy.coordinates.matrix_utilities import rotation_matrix
import astropy.units as u
from astropy.units import Quantity
from erfa import ufunc as erfa_ufunc
import numpy as np
from numpy import ndarray
import scipy.optimize as opt

from trackstream.frame.result import FrameOptimizeResult
from trackstream.stream.core import StreamArm
from trackstream.stream.plural import StreamArmsBase
from trackstream.stream.stream import Stream
from trackstream.utils.coord_utils import get_frame

__all__: list[str] = []

if TYPE_CHECKING:
    from numpy import float64
    from numpy.typing import NDArray
    from typing_extensions import TypeAlias


##############################################################################
# CODE
##############################################################################


def reference_to_skyoffset_matrix(
    lon: float | Quantity,
    lat: float | Quantity,
    rotation: float | Quantity,
) -> NDArray[float64]:
    """Convert a reference coordinate to an sky offset frame ([astropy]_).

    Cartesian to Cartesian matrix transform.

    Parameters
    ----------
    lon : float or |Angle| or |Quantity| instance
        For ICRS, the right ascension.
        If float, assumed degrees.
    lat : |Angle| or |Quantity| instance
        For ICRS, the declination.
        If float, assumed degrees.
    rotation : |Angle| or |Quantity| instance
        The final rotation of the frame about the ``origin``. The sign of
        the rotation is the left-hand rule.  That is, an object at a
        particular position angle in the un-rotated system will be sent to
        the positive |Latitude| (z) direction in the final frame.
        If float, assumed degrees.

    Returns
    -------
    ndarray
        (3x3) matrix. rotates reference Cartesian to skyoffset Cartesian.

    See Also
    --------
    :func:`~astropy.coordinates.builtin.skyoffset.reference_to_skyoffset`

    References
    ----------
    .. [astropy] Astropy Collaboration (2013).
        Astropy: A community Python package for astronomy.
        Astronomy and Astrophysics, 558, A33.
    """
    # Define rotation matrices along the position angle vector, and
    # relative to the origin.
    # None -> deg, skipping units stuff
    mat1: NDArray[float64] = rotation_matrix(-rotation, axis="x", unit=None)
    mat2: NDArray[float64] = rotation_matrix(-lat, axis="y", unit=None)
    mat3: NDArray[float64] = rotation_matrix(lon, axis="z", unit=None)

    return mat1 @ mat2 @ mat3


def residual(
    v: tuple[float, float, float],
    data: NDArray[float64],
    scalar: bool = False,  # noqa: FBT001, FBT002
) -> float64 | NDArray[float64]:
    r"""How close phi2, the rotated |Latitude| (e.g. dec), is to flat.

    This function is meant for use in a scipy minimizer.

    Parameters
    ----------
    v : tuple[float, float, float]
        (rotation, lon, lat)

        - rotation angle : float
            The final rotation of the frame about the ``origin``. The sign of
            the rotation is the left-hand rule.  That is, an object at a
            particular position angle in the un-rotated system will be sent to
            the positive |Latitude| (z) direction in the final frame.
            In degrees.
        - lon, lat : float
            In degrees. If |ICRS|, equivalent to ra & dec.

    data : (3, N) Quantity['length']
        E.g. :attr:`astropy.coordinates.ICRS.cartesian`.
    scalar : bool, optional, keyword-only
        Whether to sum `res` into a float.

    Returns
    -------
    res : float or ndarray
        :math:`\rm{lat} - 0`.
        If `scalar` is True, then sum array_like to return float.

    Other Parameters
    ----------------
    scalar : bool, optional, keyword-only
        Whether to sum `res` into a float.
        Note that if `res` is also a float, it is unaffected.
    """
    # Cartesian model
    rot_matrix = reference_to_skyoffset_matrix(v[1], v[2], v[0])
    rot_xyz: Quantity = np.dot(rot_matrix, data).T
    _, phi2 = erfa_ufunc.c2s(rot_xyz)

    res: NDArray[float64] = np.abs(phi2 - 0.0) / len(phi2)

    return np.sum(res) if scalar else res


# A slightly
# convoluted method to fit a frame -- it single-dispatches on the input type and
# then dispatches on the minimizer method. Arbitrary input types and
# minimization methods can be supported by adding to the dispatch registries.

_Rot0: TypeAlias = Quantity | float | np.floating
_default_minimize = opt.minimize


class _FitFrameNeededInfo(TypedDict):
    data: coords.BaseCoordinateFrame | coords.SkyCoord
    origin: coords.BaseCoordinateFrame | coords.SkyCoord


@functools.singledispatch
def fit_frame(
    data: object,
    rot0: _Rot0,  # noqa: ARG001
    *,
    minimizer: str | Callable[..., Any] = _default_minimize,  # noqa: ARG001
    **minimizer_kwargs: Any,  # noqa: ARG001
) -> Any:
    """Fit a frame to a set of data.

    Parameters
    ----------
    data : object
        The data to fit the frame to.
    rot0 : float or |Quantity| instance
        The initial guess for the rotation angle.
    minimizer : str or callable, optional keyword-only
        The minimizer to use. If a string, it must be a known key.
    minimizer_kwargs : dict, optional keyword-only
        Keyword arguments to pass to the minimizer.

    Returns
    -------
    result : |FrameOptimizeResult|
        The result of the fit.
    """
    msg = f"data type {type(data)} has no registered dispatch"
    raise NotImplementedError(msg)


@fit_frame.register(dict)
def _fit_frame_dict(
    info: _FitFrameNeededInfo,
    rot0: _Rot0,
    *,
    minimizer: str | Callable[..., Any] = _default_minimize,
    **minimizer_kwargs: Any,
) -> FrameOptimizeResult[Any]:
    # Get the frame from the data. Needed for the origin and result.
    from_frame = get_frame(info["data"])

    # Represent that data in unitless on-unit-sphere Cartesian coordinates.
    # Routing through UnitSphericalRepresentation to strip units and be on unit sphere.
    usrep: coords.UnitSphericalRepresentation
    usrep = info["data"].represent_as(coords.UnitSphericalRepresentation)

    crep = cast("coords.CartesianRepresentation", usrep.represent_as(coords.CartesianRepresentation))
    xyz: Quantity = crep.xyz

    # Put origin in same frame as the data.
    orep: coords.UnitSphericalRepresentation
    orep = info["origin"].transform_to(from_frame).represent_as(coords.UnitSphericalRepresentation)

    # Run minimizer
    x0 = Quantity([rot0, orep.lon, orep.lat], unit=u.deg).value
    optresult = run_minimizer(minimizer, data=xyz, x0=x0, minimizer_kwargs=minimizer_kwargs)

    # store and return minimizer results
    result: FrameOptimizeResult[Any] = FrameOptimizeResult.from_result(optresult, frame=from_frame)
    return result


@fit_frame.register(StreamArm)
def _fit_frame_StreamArm(
    stream: StreamArm,
    rot0: _Rot0,
    *,
    minimizer: str | Callable[..., Any] = _default_minimize,
    **minimizer_kwargs: Any,
) -> FrameOptimizeResult[Any]:
    """Fit Frame to a stream arm."""
    # Make dictionary of needed info
    info = _FitFrameNeededInfo(data=stream.data_coords, origin=stream.origin)
    # Fit frame
    result: FrameOptimizeResult[Any] = _fit_frame_dict(info, rot0=rot0, minimizer=minimizer, **minimizer_kwargs)

    return result


@fit_frame.register(StreamArmsBase)
def _fit_frame_StreamArmsBase(
    streams: StreamArmsBase,
    rot0: _Rot0,
    *,
    minimizer: str | Callable[..., Any] = _default_minimize,
    **minimizer_kwargs: Any,
) -> dict[str, FrameOptimizeResult[Any]]:
    """Fit Frame as many arms."""
    results = {}
    for k, v in streams.items():
        results[k] = fit_frame(v, rot0=rot0, minimizer=minimizer, **minimizer_kwargs)
    return results


@fit_frame.register(Stream)
def _fit_frame_Stream(
    stream: Stream,
    rot0: _Rot0,
    *,
    minimizer: str | Callable[..., Any] = _default_minimize,
    **minimizer_kwargs: Any,
) -> FrameOptimizeResult[Any]:
    """Fit Frame as one, not many arms."""
    # Make dictionary of needed info
    info = _FitFrameNeededInfo(data=stream.data_coords, origin=stream.origin)
    # Fit frame
    result: FrameOptimizeResult[Any] = _fit_frame_dict(info, rot0=rot0, minimizer=minimizer, **minimizer_kwargs)

    return result


# -------------------------------------------------------------------
# Minimizer dispatcher.
# Since the keys are not instances of different classes, we cannot use singledispatch
# and must roll our own equivalent using dictionaries.

_X0T = tuple[float, float, float]
_Dispatched = Callable[[ndarray, _X0T, Mapping[str, Any]], object]

MINIMIZER_DIPATCHER: dict[int, _Dispatched] = {}
# hash(name or minimizer method) -> wrapper func


def minimizer_dispatcher(key: str | Callable[..., Any]) -> Callable[[_Dispatched], _Dispatched]:
    """Register a minimizer function.

    This is a decorator factory.

    Parameters
    ----------
    key : str | Callable[..., Any]
        The name or function of the minimization method.

    Returns
    -------
    decorator : callable[[_Dispatched], _Dispatched]
        The decorator that actually registers wrapper for the minimization function.
        This decorator takes and returns the wrapper.
    """

    def decorator(method_wrapper: _Dispatched, /) -> _Dispatched:
        """Register the minimization method wrapper func to the dispatch table.

        Parameters
        ----------
        method_wrapper : _Dispatched
            The wrapper function that interfaces with the minimization method.
            The function should take the data, initial guess, and a dictionary
            of kwargs and return the minimization result.

        Returns
        -------
        method_wrapper : _Dispatched
            The unmodified input argument.
        """
        MINIMIZER_DIPATCHER[hash(key)] = method_wrapper
        return method_wrapper

    return decorator


@minimizer_dispatcher("scipy.optimize.minimize")
@minimizer_dispatcher(opt.minimize)
def scipy_optimize_minimize(
    data: NDArray[float64],
    x0: _X0T,
    minimizer_kwargs: Mapping[str, Any],
) -> opt.OptimizeResult:
    """Minimize using `scipy.optimize.minimize`.

    Parameters
    ----------
    data : ndarray
        The data to fit.
    x0 : tuple[float, float, float]
        The initial guess for the minimization.
    minimizer_kwargs : Mapping[str, Any]
        The keyword arguments to pass to `scipy.optimize.minimize`.

    Returns
    -------
    result : OptimizeResult
    """
    return opt.minimize(residual, x0=x0, args=(data, True), **minimizer_kwargs)


@minimizer_dispatcher("scipy.optimize.least_squares")
@minimizer_dispatcher(opt.least_squares)
def scipy_optimize_leastsquares(
    data: NDArray[float64],
    x0: _X0T,
    minimizer_kwargs: Mapping[str, Any],
) -> opt.OptimizeResult:
    """Minimize the residual using `scipy.optimize.least_squares`.

    Parameters
    ----------
    data : ndarray
        The data to fit.
    x0 : tuple[float, float, float]
        The initial guess for the rotation angle and origin.
    minimizer_kwargs : Mapping[str, Any]
        The keyword arguments to pass to `scipy.optimize.least_squares`.

    Returns
    -------
    result : OptimizeResult
    """
    return opt.least_squares(residual, x0=x0, args=(data, False), **minimizer_kwargs)


def run_minimizer(
    minimizer: str | Callable[..., Any],
    data: NDArray[float64],
    x0: _X0T,
    minimizer_kwargs: Mapping[str, Any],
) -> object:
    """Run the specifid minimizer on the data to find the optimal rotated frame.

    The minimizer implementation must be registered with the dispatcher
    ``MINIMIZER_DIPATCHER``.

    Parameters
    ----------
    minimizer : str or callable[..., Any]
        The minimizer method. Must be a key in the dispatcher.
    data : NDArray[float64]
        The data to finnd
    x0 : tuple[float, float, float]
        Initial position.
    minimizer_kwargs : Mapping[str, Any]
        Keyword arguments to minimizer.

    Returns
    -------
    object
        Result of minimizer.

    Raises
    ------
    NotImplementedError
        If :func:`hash` of ``minimizer`` is not in ``MINIMIZER_DIPATCHER``.
    """
    hashed = hash(minimizer)
    if hashed not in MINIMIZER_DIPATCHER:
        msg = f"minimizer {minimizer} not dispatched"
        raise NotImplementedError(msg)

    return MINIMIZER_DIPATCHER[hashed](data, x0, minimizer_kwargs)
