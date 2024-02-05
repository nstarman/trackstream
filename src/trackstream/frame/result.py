"""Fit a rotated reference frame to stream data."""

from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast, final

import astropy.coordinates as coords
import astropy.units as u
import numpy as np
from scipy.optimize import OptimizeResult

__all__: list[str] = []

if TYPE_CHECKING:
    from astropy.units import Quantity


R = TypeVar("R")


##############################################################################
# CODE
##############################################################################


@final
@dataclass(frozen=True, repr=True)
class FrameOptimizeResult(Generic[R]):
    """Result of fitting a rotated frame.

    Parameters
    ----------
    frame : |Frame|
        The fit frame.
    result : object
        Fit results, e.g. `~scipy.optimize.OptimizeResult` if the frame was fit
        using :mod:`scipy`.

    """

    frame: coords.SkyOffsetFrame
    result: R

    # ===============================================================

    @property
    def rotation(self) -> Quantity:
        """The rotation of point on sky."""
        return cast("u.Quantity", self.frame.rotation)

    @property
    def origin(self) -> coords.BaseCoordinateFrame:
        """The location of point on sky."""
        return cast("coords.BaseCoordinateFrame", self.frame.origin)

    # ===============================================================

    @singledispatchmethod
    @classmethod
    def from_result(
        cls: type[FrameOptimizeResult[Any]],
        optimize_result: object,
        frame: coords.BaseCoordinateFrame | None,
    ) -> FrameOptimizeResult[R]:
        """Construct from object.

        Parameters
        ----------
        optimize_result : object
            Instantiation is single-dispatched on the object type.
        frame : Frame | None
            The fit frame.

        Returns
        -------
        FrameOptimizeResult
            With attribute ``result`` determed by ``optimize_result``.

        Raises
        ------
        NotImplementedError
            If there is no dispatched method.
        ValueError
            If the frame is not `None` and not equal to the frame in ``optimize_result``.

        """
        if not isinstance(optimize_result, cls):
            msg = f"optimize_result type {type(optimize_result)} is not known."
            raise NotImplementedError(msg)

        # overload + Self is implemented here until it works
        if frame is not None and frame != optimize_result.frame:
            msg = "frame must be None or the same as optimize_result's frame"
            raise ValueError(msg)
        return cls(frame=optimize_result.frame, result=optimize_result.result)

    @from_result.register(OptimizeResult)
    @classmethod
    def _from_result_scipyoptresult(
        cls: type[FrameOptimizeResult[Any]],
        optimize_result: OptimizeResult,
        frame: coords.BaseCoordinateFrame,
    ) -> FrameOptimizeResult[OptimizeResult]:
        # Get coordinates
        optimize_result.x <<= u.deg
        fit_rot, fit_lon, fit_lat = optimize_result.x
        # create SkyCoord
        r = coords.UnitSphericalRepresentation(lon=fit_lon, lat=fit_lat)
        origin = coords.SkyCoord(frame.realize_frame(r, representation_type=frame.representation_type), copy=False)
        # transform to offset frame
        fit_frame = origin.skyoffset_frame(rotation=fit_rot)
        fit_frame.representation_type = frame.representation_type
        return cls(fit_frame, optimize_result)

    # ===============================================================

    def calculate_residual(self, data: coords.SkyCoord, *, scalar: bool = False) -> Quantity:
        """Calculate result residual given the fit frame.

        Parameters
        ----------
        data : (N,) `~astropy.coordinates.SkyCoord`
            The data for which to calculate the residual.
        scalar : bool
            Whether to sum the results to a scalar value.

        Returns
        -------
        Quantity
            Scalar if ``scalar``, else length N.

        """
        ur = data.transform_to(self.frame).represent_as(coords.UnitSphericalRepresentation)
        res: Quantity = np.abs(ur.lat - 0.0 * u.rad)
        return np.sum(res) if scalar else res
