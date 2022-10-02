"""Stream arm classes.

Stream arms are descriptors on a `trackstrea.Stream` class.

"""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import copy
import logging
from dataclasses import InitVar, dataclass
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar, cast

# THIRD PARTY
import numpy as np
from astropy.coordinates import BaseCoordinateFrame, SkyCoord  # noqa: TC002
from astropy.io.registry import UnifiedReadWriteMethod
from numpy.lib.recfunctions import structured_to_unstructured
from typing_extensions import TypedDict

# LOCAL
from trackstream._typing import SupportsFrame
from trackstream.clean import OUTLIER_DETECTOR_CLASSES, OutlierDetectorBase
from trackstream.io import (
    StreamArmFromFormat,
    StreamArmRead,
    StreamArmToFormat,
    StreamArmWrite,
)
from trackstream.stream.base import StreamBase
from trackstream.utils.coord_utils import f2q, get_frame, parse_framelike

if TYPE_CHECKING:
    # THIRD PARTY
    from astropy.table import QTable
    from astropy.units import Quantity
    from numpy.typing import NDArray

    # LOCAL
    from trackstream.frame import FrameOptimizeResult
    from trackstream.track.core import StreamArmTrack
    from trackstream.track.fit.core import FitterStreamArmTrack


__all__ = ["StreamArm"]


##############################################################################
# PARAMETERS


Self = TypeVar("Self", bound=SupportsFrame)  # from typing_extensions import Self


class _StreamArmCache(TypedDict):
    """Cache for Stream."""

    # frame
    frame_fit_result: FrameOptimizeResult[Any] | None
    # track
    track: StreamArmTrack | None
    track_fitter: FitterStreamArmTrack | None


##############################################################################
# CODE
##############################################################################


@dataclass(frozen=True)
class StreamArm(StreamBase):
    """An arm of a stream"""

    _CACHE_CLS: ClassVar[type] = _StreamArmCache

    data: QTable
    origin: SkyCoord
    frame: BaseCoordinateFrame | None = None
    name: str | None = None
    prior_cache: InitVar[dict[str, Any] | None] = None

    def __post_init__(self, prior_cache: dict[str, Any] | None) -> None:
        # frame-like -> frame
        object.__setattr__(self, "frame", None if self.frame is None else parse_framelike(self.frame))

        # Cache
        self._cache: dict[str, Any]
        object.__setattr__(self, "cache", prior_cache)

    read = UnifiedReadWriteMethod(StreamArmRead)
    write = UnifiedReadWriteMethod(StreamArmWrite)
    to_format = UnifiedReadWriteMethod(StreamArmToFormat)
    from_format = UnifiedReadWriteMethod(StreamArmFromFormat)

    # ===============================================================
    # Directly from Data

    def get_mask(self, minPmemb: Quantity | None = None, include_order: bool = True) -> NDArray[np.bool_]:
        """Which elements of the stream are masked."""
        minPmemb = self.flags.minPmemb if minPmemb is None else minPmemb
        mask = (self.data["Pmemb"] < minPmemb).unmasked | self.data["Pmemb"].mask
        if include_order:
            mask |= self.data["order"].mask

        return cast("np.ndarray[Any, np.dtype[np.bool_]]", mask)

    @property
    def mask(self) -> NDArray[np.bool_]:
        """Full mask: minPmemb & include_order"""
        return self.get_mask(minPmemb=None, include_order=True)

    @property
    def data_coords(self) -> SkyCoord:
        """Get ``coord`` from data table."""
        return self.data["coords"][~self.mask]

    @property
    def data_frame(self) -> BaseCoordinateFrame:
        """The `astropy.coordinates.BaseCoordinateFrame` of the data."""
        reptype = self.data["coords"].representation_type
        if not self.has_distances:
            reptype = getattr(reptype, "_unit_representation", reptype)

        # Get the frame from the data
        frame: BaseCoordinateFrame = self.data["coords"].frame.replicate_without_data(representation_type=reptype)

        return frame

    # ===============================================================
    # System stuff (fit dependent)

    def _get_order(self, mask: NDArray[np.bool_]) -> NDArray[np.int64]:
        """Get the order given a mask."""
        # original order
        iorder = self.data["order"][~mask]
        # build order
        unsorter = np.argsort(np.argsort(iorder))
        neworder = np.arange(len(iorder), dtype=int)[unsorter]

        return neworder

    @property
    def _best_frame(self) -> BaseCoordinateFrame:
        """:attr:`Stream.frame` unless its `None`, else :attr:`Stream.data_frame`."""
        frame = self.frame if self.frame is not None else self.data_frame
        return frame

    @property
    def coords(self) -> SkyCoord:
        """Data coordinates transformed to `Stream.frame` (if there is one)."""
        order = self._get_order(self.mask)

        dc = cast("SkyCoord", self.data_coords[order])
        frame = self._best_frame

        c = dc.transform_to(frame)
        c.representation_type = frame.representation_type
        c.differential_type = frame.differential_type
        return c

    # ===============================================================
    # Cleaning Data

    def mask_outliers(
        self,
        outlier_method: str | OutlierDetectorBase = "scipyKDTreeLOF",
        # kinematics: bool = False,
        *,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """Detect and label outliers, masking their Pmemb and order.

        This is done on the ``data_coords`` with minPmemb mask info.
        """
        if isinstance(outlier_method, str):
            outlier_method = OUTLIER_DETECTOR_CLASSES[outlier_method]()
        elif not isinstance(outlier_method, OutlierDetectorBase):
            raise TypeError("outlier_method must be a str or OutlierDetectorBase subclass instance")

        mask = self.get_mask(include_order=False)
        data_coords = self.data["coords"][~mask]
        # TODO! more complete, with units
        # data = structured_to_unstructured(data_coords.data._values)
        data = structured_to_unstructured(f2q(data_coords, flatten=True).value)

        # step 1: predict outlier
        isoutlier = outlier_method.fit_predict(data, data, **kwargs)

        if verbose:
            idx = np.arange(len(self.data))
            logger = logging.getLogger("trackstream")
            logger.info(f"{self.full_name} outliers: {idx[~mask][isoutlier]}")

        # step 2: set order of outliers to -1
        mask[~mask] = isoutlier
        self.data["order"][mask] = -1

        # step 3: get new order
        neworder = self._get_order(mask)
        self.data["order"][~mask] = neworder

        # step 4: remask
        self.data["order"].mask = mask

    # ===============================================================
    # Fitting Track

    @property
    def track(self) -> StreamArmTrack:
        """Stream track.

        Raises
        ------
        ValueError
            If track is not fit.
        """
        track = self.cache["track"]
        if track is None:
            raise ValueError("need to fit track. See ``arm.fit_track(...)``.")
        return track

    def fit_track(
        self,
        *,
        fitter: bool | FitterStreamArmTrack = True,
        tune: bool = True,
        force: bool = False,
        **kwargs: Any,
    ) -> StreamArmTrack:
        r"""Fit `~trackstream.fit.StreamArmTrack` to this stream.

        Parameters
        ----------
        fitter : bool or `~trackstream.fit.FitterStreamArmTrack`, optional
        keyword-only
            The fitter to use, by default `True`.
        tune : bool, optional
            Whether to train the SOM without writing to its current state
            (`True`, default). Keeping the current state is useful for re-using
            the SOM.
        force : bool, optional keyword-only
            Whether to force a track to be fit, even if one already has been for
            this stream. By default `False`.

        Returns
        -------
        `~trackstream.fit.StreamArmTrack`

        Raises
        ------
        ValueError
            If a track has already been fit and ``force`` is not `True`.
        """
        # Check if already fit
        if not force and self.cache["track"] is not None:
            raise ValueError("already fit. use ``force`` to re-fit.")

        # LOCAL
        from trackstream.track.fit.fitter import FitterStreamArmTrack

        # Get fitter instance
        if not isinstance(fitter, FitterStreamArmTrack):
            use_cached = fitter is True
            fitter = self.cache["track_fitter"]

            if not use_cached or not isinstance(fitter, FitterStreamArmTrack):
                thefitter = FitterStreamArmTrack.from_format(self)
            else:
                thefitter = fitter
        else:
            thefitter = fitter

        # if tuning, the original fitter is unaffected. This is useful for
        # saving the fitter and using it many times.
        if tune:
            thefitter = copy.deepcopy(thefitter)

        # FIT
        track = thefitter.fit(self, **kwargs)

        # Cache
        self._cache["track_fitter"] = thefitter
        self._cache["track"] = track

        # Add ordering to data table
        order = track.visit_order
        use = ~self.mask

        self.data["order"][use] = order
        self.data["order"][~use] = -1
        self.data["order"].mask = ~use

        return track

    # ===============================================================
    # Misc

    def __base_repr__(self, max_lines: int | None = None) -> list[str]:
        rs = super().__base_repr__(max_lines=max_lines)

        # 5) data table
        datarep: str = self.data._base_repr_(html=False, max_width=None, max_lines=max_lines)
        table: str = "\n\t".join(datarep.split("\n")[1:])
        rs.append("  Data:\n\t" + table)

        return rs


@get_frame.register
def _get_frame_streamarm(stream: StreamArm, /) -> BaseCoordinateFrame:
    if stream.frame is None:
        # LOCAL
        from trackstream.stream.base import FRAME_NONE_ERR

        raise FRAME_NONE_ERR

    return stream.frame
