"""Stream track fitter and fit result."""


from __future__ import annotations

import copy
from dataclasses import InitVar, dataclass
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any, TypeVar, cast, final

import astropy.coordinates as coords
import astropy.units as u
import numpy as np
from typing_extensions import ParamSpec, Self

from trackstream.stream.core import StreamArm
from trackstream.stream.plural import StreamArmsBase
from trackstream.track.fit.exceptions import (
    EXCEPT_3D_NO_DISTANCES,
    EXCEPT_NO_KINEMATICS,
)
from trackstream.track.fit.kalman.core import FirstOrderNewtonianKalmanFilter
from trackstream.track.fit.kalman.utils import make_error
from trackstream.track.fit.som import SelfOrganizingMap
from trackstream.track.fit.timesteps import make_timesteps
from trackstream.track.fit.timesteps.plural import LENGTH, SPEED, Times
from trackstream.utils.coord_utils import deep_transform_to

if TYPE_CHECKING:
    from trackstream.stream.base import StreamLike
    from trackstream.track.core import StreamArmTrack
    from trackstream.track.width.base import WidthBase
    from trackstream.track.width.plural import Widths

__all__: list[str] = []


##############################################################################
# PARAMETERS

DT = TypeVar("DT", bound=coords.BaseDifferential)
P = ParamSpec("P")


##############################################################################
# CODE
##############################################################################


# class _TSCacheDict(TypedDict):
#     """Cache for FitterStreamArmTrack."""


@final
@dataclass(frozen=True)
class FitterStreamArmTrack:
    """Track a Stream.

    When run, produces a `~trackstream.fitresult.StreamArmTrack`.

    Parameters
    ----------
    onsky : bool, keyword-only
        Should the track be fit on-sky or with distances.

    kinematics : bool or None, keyword-only
        Should the track be fit with or without kinematic information.
    """

    # The bad de
    som: SelfOrganizingMap
    """Self-Organizing Map"""
    kalman: FirstOrderNewtonianKalmanFilter
    """The Kalman Filter"""
    onsky: bool
    """Whether to fit on-sky or 3d."""
    kinematics: bool
    """Whether to fit the kinematics."""
    tocache: InitVar[dict[str, Any] | None] = None

    def __post_init__(self, tocache: dict[str, Any] | None = None) -> None:
        self._cache: dict[str, Any]
        if not hasattr(self, "_cache"):
            object.__setattr__(self, "_cache", {})
        self._cache.update(tocache or {})

        # Validate
        self._som_validator(None, self.som)
        self._kalman_validator(None, self.kalman)

    def _som_validator(self, _: Any, som: SelfOrganizingMap) -> None:
        if not isinstance(som, SelfOrganizingMap):
            raise TypeError
        if som.onsky != self.onsky:
            raise ValueError

    def _kalman_validator(self, _: Any, value: FirstOrderNewtonianKalmanFilter) -> None:
        """Validate the Kalman Filter."""
        # TODO!
        # if value.onsky != self.onsky:
        #     raise ValueError
        # if value.kinematics != self.kinematics:
        #     raise ValueError

    # ===============================================================
    # From Data

    @singledispatchmethod
    @classmethod
    def from_format(
        cls,
        arm: object,  # noqa: ARG003
        onsky: bool | None = None,  # noqa: ARG003
        kinematics: bool | None = None,  # noqa: ARG003
        *,
        som_kw: dict[str, Any] | None = None,  # noqa: ARG003
        kalman_kw: dict[str, Any] | None = None,  # noqa: ARG003
    ) -> Any:  # https://github.com/python/mypy/issues/11727
        """Create a FitterStreamArmTrack from an object."""
        msg = "not dispatched"
        raise NotImplementedError(msg)

    @from_format.register(StreamArm)
    @classmethod
    def _from_format_streamarm(  # type: ignore[misc]
        cls: type[Self],
        arm: StreamArm,
        onsky: bool | None = None,
        kinematics: bool | None = None,
        *,
        som_kw: dict[str, Any] | None = None,
        kalman_kw: dict[str, Any] | None = None,
    ) -> Self:
        if onsky is None:
            onsky = not arm.has_distances
        elif onsky is False and not arm.has_distances:
            raise EXCEPT_3D_NO_DISTANCES
        if kinematics is None:
            kinematics = arm.has_kinematics
        elif kinematics is True and not arm.has_kinematics:
            raise EXCEPT_NO_KINEMATICS

        # SOM instance
        som = SelfOrganizingMap.from_format(arm, onsky=onsky, kinematics=kinematics, **(som_kw or {}))
        # TODO? flag for kinematics

        # Kalman filter
        kalman = FirstOrderNewtonianKalmanFilter.from_format(
            arm,
            onsky=onsky,
            kinematics=kinematics,
            **(kalman_kw or {}),
        )

        return cls(som=som, kalman=kalman, onsky=onsky, kinematics=kinematics)

    @from_format.register(StreamArmsBase)
    @classmethod
    def _from_format_streamarmsbase(  # type: ignore[misc]
        cls: type[Self],
        arms: StreamArmsBase,
        onsky: bool | None = None,
        kinematics: bool | None = None,
        *,
        som_kw: dict[str, Any] | None = None,
        kalman_kw: dict[str, Any] | None = None,
    ) -> dict[str, Self]:
        return {
            n: cls.from_format(arm, onsky=onsky, kinematics=kinematics, som_kw=som_kw, kalman_kw=kalman_kw)
            for n, arm in arms.items()
        }

    # ===============================================================
    #                            FIT

    @property
    def _default_dt0(self) -> Times:
        return Times(
            {
                LENGTH: u.Quantity(0.5, u.deg) if self.kalman.onsky else u.Quantity(10.0, u.pc),
                SPEED: u.Quantity(0.01, u.Unit("mas / yr")) if self.kalman.onsky else u.Quantity(1.0, u.km / u.s),
            },
        )

    @property
    def _default_dtmin(self) -> Times:
        return Times(
            {
                LENGTH: u.Quantity(0.01, u.deg) if self.kalman.onsky else u.Quantity(0.01, u.pc),
                SPEED: u.Quantity(0.01, u.Unit("mas / yr")) if self.kalman.onsky else u.Quantity(0.01, u.km / u.s),
            },
        )

    @property
    def _default_dtmax(self) -> Times:
        return Times(
            {
                LENGTH: u.Quantity(np.inf, u.deg) if self.kalman.onsky else u.Quantity(np.inf, u.pc),
                SPEED: u.Quantity(np.inf, u.Unit("mas / yr")) if self.kalman.onsky else u.Quantity(np.inf, u.km / u.s),
            },
        )

    def fit(
        self,
        stream: StreamArm,
        *,
        som_kw: dict[str, Any] | None = None,
        kalman_kw: dict[str, Any] | None = None,
    ) -> StreamArmTrack[StreamLike]:
        """Fit a track to the data.

        Parameters
        ----------
        stream : `trackstream.Stream`, positional-only
            The stream arm to fit.

        som_kw : dict or None, optional
            Keyword options for the SOM.
        kalman_kw : dict or None, optional
            - dt0 : Times
            - dtmin : Times
            - dtmax : Times

        Returns
        -------
        StreamArmTrack instance
            Also stores as ``.track`` on the Stream
        """
        # --------------------------------------
        # Setup and Validation

        if not self.onsky and not stream.has_distances:
            raise EXCEPT_3D_NO_DISTANCES
        if self.kinematics and not stream.has_kinematics:
            raise EXCEPT_NO_KINEMATICS

        # Frame
        frame = stream.frame
        if frame is None:
            # LOCAL
            from trackstream.stream.base import FRAME_NONE_ERR

            raise FRAME_NONE_ERR

        # Get unordered arms, in frame
        data = deep_transform_to(
            stream.coords,
            frame,
            self.kalman.info.representation_type,
            self.kalman.info.differential_type,
        )

        # --------------------------------------

        # Self-Organizing Map
        projdata, order = self.som.fit_predict(data, **(som_kw or {}))
        data = data[order]  # re-order data (projdata already ordered)

        # --------------------------------------
        # Kalman Filter

        kfkw = {} if kalman_kw is None else copy.deepcopy(kalman_kw)  # decouple so can't mutate

        # Stream Width  # TODO: don't do separation from SOM, instead do
        # separation from fake prototypes, which are avg of ordered data in
        # segments.
        wb = self.som.separation(data)
        stream_width = np.convolve(wb, np.ones((10,)) / 10, mode="same")
        stream_width = cast("Widths[WidthBase]", stream_width)
        # Set minimum width from the stream width
        if (minwidth := kfkw.pop("width_min", None)) is not None:
            stream_width[stream_width < minwidth] = minwidth

        # "Timesteps", projected distances from the SOM
        # The Kalman Filter is run with `timesteps` from the SOM order.
        dt0 = Times.from_format(kfkw.pop("dt0", self._default_dt0))
        dtmin = Times.from_format(kfkw.pop("dtmin", self._default_dtmin))
        dtmax = Times.from_format(kfkw.pop("dtmax", self._default_dtmax))

        timesteps = make_timesteps(
            projdata,
            kf=self.kalman,
            dt0=dt0,
            dtmin=dtmin,
            dtmax=dtmax,
            width=kfkw.pop("width", 6),
        )
        # Always has ['length']. Also ['speed'] if there's kinematics

        # The Kalman Filter is run with widths along the stream.
        Rs = make_error(stream, self.kalman, default=0)
        Rs = cast("u.Quantity", Rs[order])

        path_name = ((stream.full_name or "") + " Path").lstrip()
        path = self.kalman.fit(data, errors=Rs, widths=stream_width, timesteps=timesteps, name=path_name)

        # -------------------
        # Return Track

        # LOCAL
        from trackstream.track.core import StreamArmTrack

        return StreamArmTrack(
            stream,
            path,
            name=stream.full_name,
            meta={"som": self.som, "visit_order": order, "kalman": self.kalman},
        )
