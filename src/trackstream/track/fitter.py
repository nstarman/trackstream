# -*- coding: utf-8 -*-

"""Stream track fitter and fit result."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union, cast

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
from numpy import apply_along_axis, array, broadcast_to, mean, ndarray, ones
from scipy.linalg import block_diag
from typing_extensions import TypedDict

# LOCAL
from .fitresult import StreamTrack
from .kalman import FirstOrderNewtonianKalmanFilter as KalmanFilter
from .kalman import kalman_output, make_Q, make_R, make_timesteps
from .path import Path, concatenate_paths
from .som import (
    CartesianSelfOrganizingMap1D,
    SelfOrganizingMap1DBase,
    UnitSphereSelfOrganizingMap1D,
)
from trackstream.utils.coord_utils import deep_transform_to

# This is to solve the circular dependency in type hint forward references # isort: skip
if TYPE_CHECKING:
    # LOCAL
    from trackstream.stream import Stream  # noqa: F401
    from trackstream.stream.arm import StreamArmDescriptor  # noqa: F401


__all__ = ["TrackStream"]


##############################################################################
# CODE
##############################################################################


class _TSCacheDict(TypedDict):
    """Cache for TrackStream."""

    frame: Optional[BaseCoordinateFrame]
    arm1_visit_order: Optional[ndarray]
    arm1_SOM: Optional[SelfOrganizingMap1DBase]
    arm1_mean_path: Optional[kalman_output]
    arm1_kalman: Optional[KalmanFilter]
    arm2_visit_order: Optional[ndarray]
    arm2_SOM: Optional[SelfOrganizingMap1DBase]
    arm2_mean_path: Optional[kalman_output]
    arm2_kalman: Optional[KalmanFilter]


class TrackStream:
    """Track a Stream.

    When run, produces a `~trackstream.fitresult.StreamTrack`.

    Parameters
    ----------
    onsky : bool, keyword-only
        Should the track be fit on-sky or with distances.

    kinematics : bool or None, keyword-only
        Should the track be fit with or without kinematic information.
    """

    # TODO! allow for resetting

    _cache: _TSCacheDict
    _onsky: bool
    _kinematics: bool

    def __init__(self, *, onsky: bool, kinematics: bool) -> None:
        self._onsky = onsky  # mutable
        self._kinematics = kinematics

        # cache starts with all `None`
        self._cache = _TSCacheDict.fromkeys(_TSCacheDict.__annotations__.keys())  # type: ignore

    @property
    def onsky(self) -> bool:
        """Whether to fit on-sky or 3d."""
        return self._onsky

    @property
    def kinematics(self) -> bool:
        """Whether to fit the kinematics."""
        return self._kinematics

    def __repr__(self) -> str:
        """String representation."""
        r = ""

        # 1) header (standard repr)
        header: str = object.__repr__(self)
        r += header

        # 2) on-sky
        r += "\n  on-sky: " + str(self.onsky)

        # 3) kinematics
        r += "\n  kinematics: " + str(self.kinematics)

        return r

    # ===============================================================
    #                            FIT
    # ===============================================================

    # ===============================================================
    # SOM

    def _fit_SOM(
        self,
        som: Optional[SelfOrganizingMap1DBase],
        stream: StreamArmDescriptor,
        data: SkyCoord,
        onsky: bool,
        tune_SOM: bool,
        kwargs: Dict[str, Any],
    ) -> SkyCoord:
        """Fit SOM to a stream arm.

        Parameters
        ----------
        stream : StreamArmDescriptor
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
        # arm info / str
        number = int(stream.name[-1])  # Get number from stream name
        arm_str = f"arm{number}"  # arm name

        if arm_str not in ("arm1", "arm2"):  # Check it's allowed.
            raise ValueError("SOM only works with 'arm1', 'arm2'.")

        vo_str = arm_str + "_visit_order"
        som_str = arm_str + "_SOM"

        # Setup
        order = None  # starts as None, will be array

        # Fit the SOM if there's data. `arm2`` doesn't always have data.
        if not stream.has_data:
            self._cache[vo_str] = order  # type: ignore
            self._cache[som_str] = som  # type: ignore
            return data
        # else:

        # 1) Try to get from cache
        if som is None:
            print("SOM is None")
            order = self._cache.get(vo_str, None)
            som = self._cache.get(som_str, None)  # type: ignore

        # 2) Fit, if None or doing further tuning
        if som is None:
            print("SOM is still None. kwargs are:", kwargs)
            order, som = self._run_SOM(data, stream.origin, onsky=onsky, som=som, **kwargs)
        else:
            if som.onsky != onsky:  # first need to check consistency of fit type.
                raise TypeError("SOM onsky doesn't match.")
            # if som.frame != frame:
            #     raise ValueError("SOM frame doesn't match")

            if tune_SOM:  # TODO! incorporate existing order
                print("SOM is tuning. kwargs are:", kwargs)
                order, som = self._run_SOM(data, stream.origin, onsky=onsky, som=som, **kwargs)

        # 3) if it's still None, give up
        if order is None:
            raise ValueError("SOM can't fit a visit order.")

        # cache
        self._cache[vo_str] = order  # type: ignore
        self._cache[som_str] = som  # type: ignore

        data = cast(SkyCoord, data[order])  # re-order

        return data

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
        random_seed: Optional[int] = None,
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

        random_seed : int or None, optional keyword-only
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
                nlattice,
                frame=frame,
                representation_type=representation_type,
                sigma=sigma,
                learning_rate=learning_rate,
                random_seed=random_seed,
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
        stream: StreamArmDescriptor,
        data: SkyCoord,
        *,
        onsky: bool,
        kinematics: bool,
        r_err: float = 0.05,
        q_err: float = 0.01,
        q_diag: float = 1,
        **options: Any,
    ) -> Union[Tuple[KalmanFilter, Path, kalman_output], Tuple[None, None, None]]:
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
        arm = f"arm{int(stream.name[-1])}"  # arm string repr for caching info

        if not stream.has_data:
            self._cache[f"{arm}_mean_path"] = None  # type: ignore
            self._cache[f"{arm}_kalman"] = None  # type: ignore

            return None, None, None

        # -------------------
        # Get the Frame and Representation(+Differential)

        representation_type = UnitSphericalRepresentation if onsky else CartesianRepresentation
        nrepdims: int = 2 if onsky else 3

        if not kinematics:
            differential_type = None
        elif onsky:  # TODO! need to be cognizant if just radial, etc.
            differential_type = UnitSphericalDifferential
        else:
            differential_type = CartesianDifferential

        # frame, with rep-type set
        frame = data.frame.replicate_without_data(
            representation_type=representation_type,
            differential_type=differential_type,
        )

        ndifdims: int
        if kinematics:
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

        if onsky:
            dt0 = Quantity(0.5, u.deg)
            vmin = Quantity(0.01, u.deg)
        else:
            dt0 = Quantity(50, u.pc)
            vmin = Quantity(0.01, u.pc)
        timesteps = make_timesteps(data, dt0=dt0, width=6, vmin=vmin)

        # -------------------
        # Fit

        kf = KalmanFilter(x0, P0, onsky=onsky, kinematics=kinematics, frame=frame, Q0=Q, **options)
        path, center = kf.fit(data, R, timesteps=timesteps, name=stream.name)

        # Cache info
        self._cache[f"{arm}_mean_path"] = center  # type: ignore
        self._cache[f"{arm}_kalman"] = kf  # type: ignore

        return kf, path, center

    # -------------------------------------------

    def fit(
        self,
        stream: Stream,  # type: ignore
        /,
        *,
        force: bool = False,  # TODO!
        arm1SOM: Optional[SelfOrganizingMap1DBase] = None,
        arm2SOM: Optional[SelfOrganizingMap1DBase] = None,
        tune_SOM: bool = True,
        som_fit_kw: Optional[dict] = None,
        kalman_fit_kw: Optional[dict] = None,
    ) -> StreamTrack:
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

        arm1SOM, arm2SOM : `~trackstream.SelfOrganizingMap1DBase` or None, optional keyword-only
            Fiducial SOMs for stream arms 1 and 2, respectively. Warning: the
            attribute ``onsky`` will be mutated in-place to `onsky`.
        tune_SOM : bool, optional keyword-only
        som_fit_kw : dict or None, optional keyword-only

        kalman_fit_kw : dict or None, optional keyword-only

        Returns
        -------
        StreamTrack instance
            Also stores as ``.track`` on the Stream
        """
        # --------------------------------------
        # Validation

        # Onsky
        onsky = self.onsky
        if not onsky and not stream.has_distances:
            raise ValueError(
                "this stream does not have distance information; "
                "cannot compute track with distances.",
            )

        # Kinematics.
        kinematics = self.kinematics
        if kinematics and not stream.has_kinematics:  # check they exist
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

        representation_type = frame.representation_type

        # Get unordered arms, in frame
        data1 = deep_transform_to(
            stream.arm1.coords,
            frame,
            representation_type,
            differential_type=None,
        )
        data2 = deep_transform_to(
            stream.arm2.coords,
            frame,
            representation_type,
            differential_type=None,
        )

        # --------------------------------------
        # Self-Organizing Map
        # Unlike the previous step, we must do this for both arms.

        # SOM
        if arm1SOM is not None and (onsky is not None and arm1SOM.onsky != onsky):
            raise ValueError
        elif arm2SOM is not None and (onsky is not None and arm2SOM.onsky != onsky):
            raise ValueError

        som_kw = som_fit_kw or {}  # None -> dict
        som_kw.setdefault("maxsep", Quantity(5, u.deg) if onsky else None)

        # Arm 1
        data1 = self._fit_SOM(
            arm1SOM,
            stream.arm1,
            data1,
            onsky=onsky,
            tune_SOM=tune_SOM,
            kwargs=som_kw,
        )

        # Arm 2 (if not None)
        data2 = self._fit_SOM(
            arm2SOM,
            stream.arm2,
            data2,
            onsky=onsky,
            tune_SOM=tune_SOM,
            kwargs=som_kw,
        )

        # -------------------
        # Kalman Filter
        # both arms start at 0 displacement wrt themselves, but not each other.
        # e.g. the progenitor is cut out. To address this the start of affine
        # is offset by epsilon = min(1e-10, 1e-10 * dp2p[0])

        kf_kw: Dict[str, Any] = kalman_fit_kw or {}

        # Arm 1  (never None)
        kf1, path1, _ = self._fit_kalman_filter(
            stream.arm1, data1, onsky=onsky, kinematics=kinematics, **kf_kw
        )
        # Arm 2
        kf2, path2, _ = self._fit_kalman_filter(
            stream.arm2, data2, onsky=onsky, kinematics=kinematics, **kf_kw
        )

        # -------------------
        # Combine together into a single path
        # Need to reverse order of one arm to be indexed toward origin, not away

        path_name = ((stream.full_name or "") + " Path").lstrip()

        if path1 is None:
            raise ValueError
        elif path2 is not None:
            # TODO! which is the negative?
            path = concatenate_paths((path2, path1), name=path_name)
        else:
            path = path1
            path._name = path_name  # Rename, since `_fit_...` does not

        # construct interpolation
        track = StreamTrack(
            stream,
            path,
            origin=stream.origin,
            name=stream.full_name,
            # metadata
            # frame_fit=frame_fit,
            som=dict(arm1=self._cache.get("arm1_SOM"), arm2=self._cache.get("arm2_SOM")),
            kalman=dict(arm1=kf1, arm2=kf2),
            meta=None,
        )
        return track
