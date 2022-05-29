# -*- coding: utf-8 -*-

"""Stream track fitter and fit result."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, cast

# THIRD PARTY
import astropy.units as u
from astropy.coordinates import (
    BaseCoordinateFrame,
    CartesianDifferential,
    CartesianRepresentation,
    SkyCoord,
    UnitSphericalDifferential,
    UnitSphericalRepresentation,
)
from astropy.units import Quantity
from attrs import NOTHING, define, field
from numpy import apply_along_axis, array, broadcast_to, mean, ndarray, ones
from scipy.linalg import block_diag
from typing_extensions import TypedDict

# LOCAL
from .kalman import FirstOrderNewtonianKalmanFilter as KalmanFilter
from .kalman import kalman_output, make_Q, make_R, make_timesteps
from .path import Path
from .som import (
    CartesianSelfOrganizingMap1D,
    SelfOrganizingMap1DBase,
    UnitSphereSelfOrganizingMap1D,
)
from .track import StreamArmTrack
from trackstream.utils._attrs import (
    _cache_factory,
    _cache_proxy_factory,
    convert_if_none,
)
from trackstream.utils.coord_utils import deep_transform_to

# This is to solve the circular dependency in type hint forward references # isort: skip
if TYPE_CHECKING:
    # LOCAL
    from trackstream.stream.core import StreamArm  # noqa: F401


__all__ = ["TrackStreamArm"]


##############################################################################
# CODE
##############################################################################


class _TSCacheDict(TypedDict):
    """Cache for TrackStreamArm."""

    frame: Optional[BaseCoordinateFrame]
    som: Optional[SelfOrganizingMap1DBase]
    visit_order: Optional[ndarray]
    kalman: Optional[KalmanFilter]
    mean_path: Optional[kalman_output]


@define(frozen=True, kw_only=True)
class TrackStreamArm:
    """Track a Stream.

    When run, produces a `~trackstream.fitresult.StreamArmTrack`.

    Parameters
    ----------
    onsky : bool, keyword-only
        Should the track be fit on-sky or with distances.

    kinematics : bool or None, keyword-only
        Should the track be fit with or without kinematic information.
    """

    onsky: bool
    """Whether to fit on-sky or 3d."""

    kinematics: bool
    """Whether to fit the kinematics."""

    _cache: dict = field(
        kw_only=True,
        factory=_cache_factory(_TSCacheDict),
        converter=convert_if_none(_cache_factory(_TSCacheDict), deepcopy=True),
    )
    cache: MappingProxyType = field(init=False, default=_cache_proxy_factory)

    # ===============================================================
    #                            FIT
    # ===============================================================

    # ===============================================================
    # SOM

    def _fit_SOM(
        self,
        som: Optional[SelfOrganizingMap1DBase],
        stream: StreamArm,
        data: SkyCoord,
        onsky: bool,
        tune_SOM: bool,
        kwargs: Dict[str, Any],
    ) -> Tuple[SelfOrganizingMap1DBase, SkyCoord, ndarray]:
        """Fit SOM to a stream arm.

        Parameters
        ----------
        stream : StreamArm
            The stream arm instance, with information like the stream's origin.
        data : SkyCoord
            The data.
        number : int
            Which stream arm.
        onsky : bool
            Whether to fit in 3D or on they sky (a unit sphere)
        tune_SOM : bool
            Whether to run the SOM even if it's already been fit.
            This is useful if a pre-trained SOM is given.
        kwargs : dict
            Keyword arguments for ``_run_SOM``.

        Returns
        -------
        SkyCoord
            The data, ordered.

        Raises
        ------
        TypeError
            If the onsky of the SOM is not the same as the ``onsky`` argument.
            This can only happen if a pre-trained SOM is given.
        ValueError
            If the som's frame is not the same as the ``frame`` argument.
            This can only happen if a pre-trained SOM is given.
        ValueError
            If the SOM fails miserably and cannot create a visit_order.
        """
        # Setup
        order = None  # starts as None, will be array

        # 1) Try to get from cache
        if som is None:
            print("SOM is None")
            order = self.cache["visit_order"]
            som = self.cache["som"]

        # 2) Fit, if None or doing further tuning
        if som is None:
            print("SOM is still None. kwargs are:", kwargs)
            order, som = self._run_SOM(data, stream.origin, onsky=onsky, som=som, **kwargs)
        else:
            if som.onsky != onsky:  # first need to check consistency of fit type.
                raise TypeError("SOM onsky doesn't match.")

            if tune_SOM:  # TODO! incorporate existing order
                print("SOM is tuning. kwargs are:", kwargs)
                order, som = self._run_SOM(data, stream.origin, onsky=onsky, som=som, **kwargs)

        # 3) if it's still None, give up
        if order is None:
            raise ValueError("SOM can't fit a visit order.")

        # cache
        self._cache["visit_order"] = order
        self._cache["som"] = som

        data = cast(SkyCoord, data[order])  # re-order
        return som, data, order

    def _run_SOM(
        self,
        data: SkyCoord,
        origin: SkyCoord,
        onsky: bool,
        *,
        som: Optional[SelfOrganizingMap1DBase] = None,
        learning_rate: float = 0.1,
        sigma: float = 1.0,
        num_iteration: int = 10_000,
        rng: Optional[int] = None,
        progress: bool = False,
        nlattice: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[ndarray, SelfOrganizingMap1DBase]:
        """Reorder data by SOM.

        Parameters
        ----------
        data : SkyCoord
        origin : SkyCoord
        onsky : bool
            Whether the track should be fit on-sky or with distances.

        som : object or None, optional
            The self-organizing map. If None, will be constructed.
        learning_rate : float, optional keyword-only
        sigma : float, optional keyword-only
        num_iteration : int, optional keyword-only

        rng : int or None, optional keyword-only
        progress : bool, optional keyword-only
            Whether to show progress bar.

        Returns
        -------
        visit_order : array
        som : SelfOrganizingMap1D

        Other Parameters
        ----------------
        nlattice : int or None, optional keyword-only
            Number of lattice (prototype) points.
        maxsep : Quantity or None, optional keyword-only
            Maximum separation (in data space) between prototypes.
        """
        # The SOM
        if som is None:
            # The SOM class
            SOM = UnitSphereSelfOrganizingMap1D if onsky else CartesianSelfOrganizingMap1D

            # The Frame and Rep-type in which to do the SOM
            frame = data.frame.replicate_without_data()
            representation_type = data.frame.representation_type

            # Determine number of lattice points, within [5, 100]
            nlattice = min(max(len(data) // 50, 5), 100) if nlattice is None else nlattice

            # Instantiate SOM
            som = SOM(
                nlattice=nlattice,
                frame=frame,
                frame_representation_type=representation_type,
                frame_differential_type=None,
                sigma=sigma,
                learning_rate=learning_rate,
                rng=rng,  # type: ignore
                prototypes=NOTHING,  # type: ignore
            )
            # And initial weights
            som.make_prototypes_binned(data, **kwargs)

        # get the ordering by "vote" of the Prototypes
        visit_order = som.fit_predict(
            data,
            num_iteration=num_iteration,
            random_order=False,
            progress=progress,
            origin=origin,
        )

        return visit_order, som

    # FIXME! r_err
    def _fit_kalman_filter(
        self,
        stream: StreamArm,
        data: SkyCoord,
        *,
        r_err: float = 0.05,
        q_err: float = 0.01,
        q_diag: float = 1,
        **options: Any,
    ) -> Tuple[KalmanFilter, Path]:
        """Fit data with Kalman filter.

        Parameters
        ----------
        data : SkyCoord
            In frame ``frame`` with ``representation_type``
        frame : BaseCoordinateFrame

        onsky : bool
        kinematics : bool

        Returns
        -------
        mean_path
        kalman_filter
        """
        # -------------------
        # Get the Frame and Representation(+Differential)

        representation_type = UnitSphericalRepresentation if self.onsky else CartesianRepresentation
        nrepdims: int = 2 if self.onsky else 3

        if not self.kinematics:
            differential_type = None
        elif self.onsky:  # TODO! need to be cognizant if just radial, etc.
            differential_type = UnitSphericalDifferential
        else:
            differential_type = CartesianDifferential

        # frame, with rep-type set
        frame = data.frame.replicate_without_data(
            representation_type=representation_type,
            differential_type=differential_type,
        )

        ndifdims: int
        if self.kinematics:
            data = deep_transform_to(
                data,
                frame,
                representation_type,
                differential_type,
            )  # TODO necessary?
            ndifdims = len(frame.get_representation_component_names("s"))
        else:  # strip kinematics
            rep = data.data.without_differentials()
            data = SkyCoord(
                data.realize_frame(rep),
                representation_type=representation_type,
                copy=False,
            )
            ndifdims = 0

        ndims = nrepdims + ndifdims

        # -------------------
        # Initial Conditions

        # starting position
        x0 = options.get("x0", None)
        if x0 is None:  # need to determine a good starting point
            # Instead of choosing the first point as the starting point,
            # since the stream is in its frame, instead choose the locus of
            # points near the origin.
            r = data.represent_as(representation_type, s=differential_type)
            x0 = r[:3].without_differentials().mean()

            if "s" in r.differentials:  # can't take mean if there aren't differentials
                # use a hack to get the mean of the
                drmean = apply_along_axis(mean, 0, r.differentials["s"][:3]).item()
                x0.differentials["s"] = drmean
            # else: make 0 diff if there aren't errors

        # TODO! as options
        p = array([[0.0001, 0], [0, 1]])
        P0 = block_diag(*(p,) * ndims)

        # TODO! actual errors
        err = broadcast_to(r_err, ndims)
        errs = ones((len(data), ndims)) * err[None]
        R = make_R(errs)  # TODO! actual errors
        print(R[0])

        options["q_kw"] = dict(var=q_err, diag=q_diag)  # TODO! overridable

        def Q(dt: float, var: float = 1, diag: float = 1, ndims: int = 3) -> ndarray:
            return make_Q(dt, var, ndims=ndims) + block_diag(*[array([[diag, 0], [0, 0]])] * ndims)

        # -------------------
        # Time steps

        if self.onsky:
            dt0 = Quantity(0.5, u.deg)
            vmin = Quantity(0.01, u.deg)
        else:
            dt0 = Quantity(50, u.pc)
            vmin = Quantity(0.01, u.pc)
        timesteps = make_timesteps(data, dt0=dt0, width=6, vmin=vmin)

        # -------------------
        # Fit

        kf = KalmanFilter.from_representation(
            x0,
            P0,
            onsky=self.onsky,
            kinematics=self.kinematics,
            frame=frame,
            Q0=None,
            process_noise_model=Q,
            **options,
        )
        path_name = ((stream.full_name or "") + " Path").lstrip()
        path = kf.fit(data, R, timesteps=timesteps, name=path_name)

        # Cache info
        self._cache["kalman"] = kf
        self._cache["mean_path"] = path

        return kf, path

    # -------------------------------------------

    def fit(
        self,
        stream: StreamArm,
        /,
        *,
        force: bool = False,  # TODO!
        som: Optional[SelfOrganizingMap1DBase] = None,
        tune_SOM: bool = True,
        som_fit_kw: Optional[dict] = None,
        kalman_fit_kw: Optional[dict] = None,
    ) -> StreamArmTrack:
        """Fit a track to the data.

        Parameters
        ----------
        stream : `trackstream.Stream`, positional-only onsky : bool or None,
        optional
            Should the track be fit by on-sky or with distances. If `None`
            (default) the data is inspected to see if it has distances.
        kinematics : bool or None, optional keyword-only
            Should the track be fit with or without kinematics? If `None`
            (default) the data is inspected to see if it has kinematic
            information.

        force : bool
            Whether to force a refit from scratch.

        frame_fit_kw : dict or None, optional keyword-only

        som : `~trackstream.SelfOrganizingMap1DBase` or None, optional keyword-only
            Fiducial SOM for stream arm.
        tune_SOM : bool, optional keyword-only
        som_fit_kw : dict or None, optional keyword-only

        kalman_fit_kw : dict or None, optional keyword-only

        Returns
        -------
        StreamArmTrack instance
            Also stores as ``.track`` on the Stream
        """
        # --------------------------------------
        # Validation

        # Onsky
        if not self.onsky and not stream.has_distances:
            raise ValueError(
                "this stream does not have distance information; "
                "cannot compute track with distances.",
            )

        # Kinematics.
        if self.kinematics and not stream.has_kinematics:  # check they exist
            raise ValueError(
                "this stream does not have kinematic information; "
                "cannot compute track with velocities.",
            )

        # Frame
        frame = stream.system_frame
        # NOT ._init_system_frame, to pick up on fit frames
        if frame is None:
            raise ValueError(
                "cannot fit a track without a system frame. " "see ``Stream.fit_frame``.",
            )

        # --------------------------------------
        # Setup

        # Get unordered arms, in frame
        data = deep_transform_to(
            stream.coords,
            frame,
            frame.representation_type,
            differential_type=None if not self.kinematics else frame.differential_type,
        )

        # --------------------------------------
        # Self-Organizing Map

        # SOM
        if som is not None and som.onsky != self.onsky:
            raise ValueError

        som_kw = som_fit_kw or {}  # None -> dict
        som_kw.setdefault("maxsep", Quantity(5, u.deg) if self.onsky else None)

        # Arm 1
        som, data, visit_order = self._fit_SOM(
            som,
            stream,
            data,
            onsky=self.onsky,
            tune_SOM=tune_SOM,
            kwargs=som_kw,
        )

        # -------------------
        # Kalman Filter
        # arms start at 0 displacement wrt themselves, but not each other.
        # e.g. the progenitor is cut out. To address this the start of affine
        # is offset by epsilon = min(1e-10, 1e-10 * dp2p[0])

        kf_kw: Dict[str, Any] = kalman_fit_kw or {}
        kf, path = self._fit_kalman_filter(stream, data, **kf_kw)

        # -------------------
        # Make Track

        track = StreamArmTrack(
            stream,
            path,
            name=stream.full_name,
            # metadata
            meta=dict(som=som, visit_order=visit_order, kalman=kf),  # type: ignore
        )
        return track
