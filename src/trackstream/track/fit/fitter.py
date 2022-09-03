"""Stream track fitter and fit result."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from dataclasses import InitVar, dataclass
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any, ClassVar, NoReturn, TypeVar, cast, final

# THIRD PARTY
import astropy.coordinates as coords
import astropy.units as u
import numpy as np
from numpy import ndarray
from typing_extensions import ParamSpec, Self, TypedDict

# LOCAL
from trackstream.stream.base import CacheProperty
from trackstream.stream.core import StreamArm
from trackstream.stream.plural import StreamArmsBase
from trackstream.track.fit.errors import EXCEPT_3D_NO_DISTANCES, EXCEPT_NO_KINEMATICS
from trackstream.track.fit.kalman.base import kalman_output
from trackstream.track.fit.kalman.core import FirstOrderNewtonianKalmanFilter
from trackstream.track.fit.kalman.utils import make_error
from trackstream.track.fit.som import SelfOrganizingMap
from trackstream.track.fit.timesteps import make_timesteps
from trackstream.track.fit.timesteps.plural import Times
from trackstream.track.width.plural import Widths
from trackstream.utils.coord_utils import deep_transform_to

if TYPE_CHECKING:
    # LOCAL
    from trackstream.track.core import StreamArmTrack

__all__ = ["FitterStreamArmTrack"]


##############################################################################
# PARAMETERS

DT = TypeVar("DT", bound=coords.BaseDifferential)
P = ParamSpec("P")


##############################################################################
# CODE
##############################################################################


class _TSCacheDict(TypedDict):
    """Cache for FitterStreamArmTrack."""

    visit_order: ndarray | None
    mean_path: kalman_output | None


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

    _CACHE_CLS: ClassVar[type] = _TSCacheDict
    cache = CacheProperty()

    # The bad de
    som: SelfOrganizingMap
    """Self-Organizing Map"""
    kalman: FirstOrderNewtonianKalmanFilter
    """The Kalman Filter"""
    # _: KW_ONLY
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

        elif som.onsky != self.onsky:
            raise ValueError

    def _kalman_validator(self, _: Any, value: FirstOrderNewtonianKalmanFilter) -> None:
        """TODO!"""
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
        arm: object,
        onsky: bool | None = None,
        kinematics: bool | None = None,
        *,
        som_kw: dict[str, Any] | None = None,
        kalman_kw: dict | None = None,
    ) -> NoReturn:
        raise NotImplementedError("not dispatched")

    @from_format.register(StreamArm)
    @classmethod
    def _from_format_streamarm(
        cls: type[Self],
        arm: StreamArm,
        onsky: bool | None = None,
        kinematics: bool | None = None,
        *,
        som_kw: dict[str, Any] | None = None,
        kalman_kw: dict | None = None,
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
            arm, onsky=onsky, kinematics=kinematics, **(kalman_kw or {})
        )

        return cls(som=som, kalman=kalman, onsky=onsky, kinematics=kinematics)

    @from_format.register(StreamArmsBase)
    @classmethod
    def _from_format_streamarmsbase(
        cls: type[Self],
        arms: StreamArmsBase,
        onsky: bool | None = None,
        kinematics: bool | None = None,
        *,
        som_kw: dict[str, Any] | None = None,
        kalman_kw: dict | None = None,
    ) -> dict[str, Self]:
        return {
            n: cls.from_format(arm, onsky=onsky, kinematics=kinematics, som_kw=som_kw, kalman_kw=kalman_kw)
            for n, arm in arms.items()
        }

    # ===============================================================
    #                            FIT

    @property
    def _default_dt0(self) -> Times:
        dt0 = Times(
            {
                "length": u.Quantity(0.5, u.deg) if self.kalman.onsky else u.Quantity(10.0, u.pc),
                "speed": u.Quantity(0.01, u.Unit("mas / yr")) if self.kalman.onsky else u.Quantity(1.0, u.km / u.s),
            }
        )
        return dt0

    @property
    def _default_dtmin(self) -> Times:
        dtmin = Times(
            {
                "length": u.Quantity(0.01, u.deg) if self.kalman.onsky else u.Quantity(0.01, u.pc),
                "speed": u.Quantity(0.01, u.Unit("mas / yr")) if self.kalman.onsky else u.Quantity(0.01, u.km / u.s),
            }
        )
        return dtmin

    def fit(
        self,
        stream: StreamArm,
        *,
        som_kw: dict | None = None,
        kalman_kw: dict | None = None,
    ) -> StreamArmTrack:
        """Fit a track to the data.

        Parameters
        ----------
        stream : `trackstream.Stream`, positional-only
            The stream arm to fit.

        width0 : Widths
        som_kw : dict or None, optional keyword-only
            Keyword options for the SOM.
        kalman_kw : dict or None, optional keyword-only
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

        onsky = self.onsky
        kinematics = self.kinematics

        if not onsky and not stream.has_distances:
            raise EXCEPT_3D_NO_DISTANCES
        elif kinematics and not stream.has_kinematics:
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
        self._cache["visit_order"] = order  # cache result
        data = data[order]  # re-order data

        # Stream Width
        wb = self.som.separation(data)
        stream_width = np.convolve(wb, np.ones((10,)) / 10, mode="same")  # TODO! better kernel width selection
        stream_width = cast(Widths, stream_width)

        # "Timesteps", projected distances from the SOM
        # The Kalman Filter is run with `timesteps` from the SOM order.
        # Options are packaged in the kalman filter's kwargs
        kfkw = {} if kalman_kw is None else kalman_kw

        dt0 = Times.from_format(kfkw.pop("dt0", self._default_dt0))
        dtmin = Times.from_format(kfkw.pop("dtmin", self._default_dtmin))
        _dtmax = kfkw.pop("dtmax", None)
        dtmax = Times.from_format(_dtmax) if _dtmax is not None else None

        timesteps = make_timesteps(projdata, kf=self.kalman, dt0=dt0, dtmin=dtmin, dtmax=dtmax, width=6)

        # # THIRD PARTY
        # import matplotlib.pyplot as plt

        # fig = plt.figure()
        # plt.plot(timesteps["length"].value)
        # plt.show()

        # Kalman Filter
        # The Kalman Filter is run with widths along the stream.
        Rs = make_error(stream, self.kalman, default=0)
        Rs = cast(u.Quantity, Rs[order])

        path_name = ((stream.full_name or "") + " Path").lstrip()
        path = self.kalman.fit(data, errors=Rs, widths=stream_width, timesteps=timesteps, name=path_name)

        # Cache result
        self._cache["mean_path"] = path

        # -------------------
        # Return Track

        # LOCAL
        from trackstream.track.core import StreamArmTrack

        track = StreamArmTrack(
            stream, path, name=stream.full_name, meta=dict(som=self.som, visit_order=order, kalman=self.kalman)
        )

        return track
