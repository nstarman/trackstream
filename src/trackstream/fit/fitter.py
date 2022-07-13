"""Stream track fitter and fit result."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Type, cast

# THIRD PARTY
import astropy.units as u
import numpy as np
from astropy.coordinates import BaseDifferential as BaseDif
from astropy.coordinates import SkyCoord
from astropy.units import Quantity
from attrs import define, field
from numpy import ndarray
from typing_extensions import Self, TypedDict

# LOCAL
from .errors import EXCEPT_3D_NO_DISTANCES, EXCEPT_NO_KINEMATICS
from .kalman import FirstOrderNewtonianKalmanFilter as KalmanFilter
from .kalman import kalman_output, make_error
from .som import (
    CartesianSelfOrganizingMap1D,
    SelfOrganizingMap1DBase,
    UnitSphereSelfOrganizingMap1D,
)
from .timestep import make_timesteps
from .track import StreamArmTrack
from trackstream.fit.width import make_stream_width
from trackstream.utils._attrs import _cache_factory, convert_if_none
from trackstream.utils.coord_utils import deep_transform_to

if TYPE_CHECKING:
    # LOCAL
    from trackstream.stream.core import StreamArm  # noqa: F401


__all__ = ["FitterStreamArmTrack"]


##############################################################################
# CODE
##############################################################################


@define(frozen=True, kw_only=True)
class FitterStreamArmTrackBase:
    """Base class for fitters to a `trackstream.stream.StreamArm`.

    Currently the only available fitter is `FitterStreamArmTrack`.

    Parameters
    ----------
    onsky : bool
    kinematics : bool
    _cache : dict
    """

    onsky: bool = field(default=True, kw_only=True, converter=bool)
    """Whether to fit on-sky or 3d."""

    kinematics: bool = field(default=True, kw_only=True, converter=bool)
    """Whether to fit the kinematics."""

    _cache: dict = field(kw_only=True, factory=dict, converter=convert_if_none(dict, deepcopy=True))


##############################################################################


class _TSCacheDict(TypedDict):
    """Cache for FitterStreamArmTrack."""

    # frame: Optional[BaseCoordinateFrame]
    visit_order: ndarray | None
    mean_path: kalman_output | None


@define(frozen=True, kw_only=True)
class FitterStreamArmTrack(FitterStreamArmTrackBase):
    """Track a Stream.

    When run, produces a `~trackstream.fitresult.StreamArmTrack`.

    Parameters
    ----------
    onsky : bool, keyword-only
        Should the track be fit on-sky or with distances.

    kinematics : bool or None, keyword-only
        Should the track be fit with or without kinematic information.
    """

    som: SelfOrganizingMap1DBase = field()
    """Self-Organizing Map"""

    kalman: KalmanFilter = field()
    """The Kalman Filter"""

    _cache: dict = field(
        kw_only=True,
        factory=_cache_factory(_TSCacheDict),
        converter=convert_if_none(_cache_factory(_TSCacheDict), deepcopy=True),
        repr=False,
    )

    @som.validator  # type: ignore
    def _som_validator(self, _, value: SelfOrganizingMap1DBase):
        if value.onsky != self.onsky:
            raise ValueError

    @kalman.validator  # type: ignore
    def _kalman_validator(self, _, value: KalmanFilter):
        if value.onsky != self.onsky:
            raise ValueError
        if value.kinematics != self.kinematics:
            raise ValueError

    @property
    def cache(self) -> MappingProxyType:
        return MappingProxyType(self._cache)

    # ===============================================================
    # From Data

    @classmethod
    def from_stream(
        cls: type[Self],
        arm: StreamArm,
        /,
        onsky: bool | None = None,
        kinematics: bool | None = None,
        *,
        som_kw: dict[str, Any] | None = None,
        som_bin_kw: dict[str, Any] | None = None,
        kalman_kw: dict | None = None,
        kdtree_kw: dict | None = None,
    ) -> Self:
        # -----------------------
        # Determine/check onsky/kinematics

        if onsky is None:
            onsky = not arm.has_distances
        elif onsky is False and not arm.has_distances:
            raise EXCEPT_3D_NO_DISTANCES
        if kinematics is None:
            kinematics = arm.has_kinematics
        elif kinematics is True and not arm.has_kinematics:
            raise EXCEPT_NO_KINEMATICS

        # -----------------------

        # The SOM class
        SOM = UnitSphereSelfOrganizingMap1D if onsky else CartesianSelfOrganizingMap1D
        # SOM instance
        som = SOM.from_stream(arm, **(som_kw or {}))
        # And initial weights on unordered coordinates
        som.make_prototypes_binned(arm.coords_ord, **(som_bin_kw or {}))

        # Kalman filter
        kalman = KalmanFilter.from_stream(arm, onsky=onsky, kinematics=kinematics, **(kalman_kw or {}))

        return cls(som=som, kalman=kalman, onsky=onsky, kinematics=kinematics)

    # ===============================================================
    #                            FIT

    def fit(
        self,
        stream: StreamArm,
        /,
        stream_width: Quantity,
        *,
        minPmemb: Quantity[u.percent] = Quantity(80, u.percent),
        som_kw: dict | None = None,
        kalman_kw: dict | None = None,
    ) -> StreamArmTrack:
        """Fit a track to the data.

        Parameters
        ----------
        stream : `trackstream.Stream`, positional-only
            The stream arm to fit.

        stream_width : Quantity
            A structured quantity of the stream width.
            Field names must match the frame or representation.

        som_kw : dict or None, optional keyword-only
            Keyword options for the SOM.
        kalman_kw : dict or None, optional keyword-only

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
        frame = stream.system_frame  # NOT ._init_system_frame, b/c use fit frames
        if frame is None:
            msg = "cannot fit a track without a system frame (see ``Stream.fit_frame``)."
            raise ValueError(msg)

        # Get the data that should be used in the track
        # 1) high probability
        # 2) positive order (basically marked later as having high probability)
        use = (stream.data["Pmemb"] >= minPmemb) & (stream.data["order"] >= 0)

        # Get unordered arms, in frame
        fdt = cast(Type[BaseDif], frame.differential_type)
        data = deep_transform_to(
            cast(SkyCoord, stream.coords[use]),
            frame,
            frame.representation_type,
            differential_type=None if not kinematics else fdt,
        )

        # --------------------------------------

        # Self-Organizing Map
        projdata, order = self.som.fit_predict(data, origin=stream.origin, **(som_kw or {}))

        # return (data, order, self.som)

        data = cast(SkyCoord, data[order])  # re-order data

        self._cache["visit_order"] = order  # cache result

        # Stream Width
        # TODO! more robust. This quick and dirty method doesn't do the velocities
        stream_width = make_stream_width(data, self.som, stream_width)

        # THIRD PARTY
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(stream_width["x"])

        # Timesteps from the SOM
        # The Kalman Filter is run with `timesteps` from the SOM order.
        # Options are packaged in the kalman filter's kwargs
        kalman_kw = {} if kalman_kw is None else kalman_kw
        fields = ("positions", "kinematics") if kinematics else ("positions",)
        dtype = np.dtype([(n, float) for n in fields])
        units = (u.deg if onsky else u.pc,) + ((u.Unit("mas/yr") if onsky else u.km / u.s,) if kinematics else ())

        dt0 = kalman_kw.pop("dt0", None)
        dtmin = kalman_kw.pop("dtmin", None)
        dtmax = kalman_kw.pop("dtmax", np.inf)
        if dt0 is None:
            dt0 = Quantity(
                (0.5 if onsky else 10,) + ((0.01 if onsky else 1,) if kinematics else ()),
                unit=units,
                dtype=dtype,
            )
        if dtmin is None:
            dtmin = Quantity((0.01,) + ((0.01,) if kinematics else ()), unit=units, dtype=dtype)

        # TODO! include kinematics in projection
        projdata.data.differentials.update(data.data.differentials)
        timesteps = make_timesteps(
            projdata, onsky=onsky, kinematics=kinematics, dt0=dt0, width=6, dtmin=dtmin, dtmax=dtmax
        )

        # THIRD PARTY
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(timesteps["positions"])

        # Kalman Filter
        # The Kalman Filter is run with widths along the stream.
        Rs = make_error(self.kalman, stream.data, default=0)
        Rs = cast(Quantity, Rs[order])  # todo? instead do stream.data[order]

        path_name = ((stream.full_name or "") + " Path").lstrip()
        path = self.kalman.fit(data, errors=Rs, widths=stream_width, timesteps=timesteps, name=path_name)

        # Cache result
        self._cache["mean_path"] = path

        # -------------------
        # Return Track

        track = StreamArmTrack(
            stream,
            path,
            name=stream.full_name,
            # metadata
            meta=dict(som=self.som, visit_order=order, kalman=self.kalman),  # type: ignore
        )
        return track
