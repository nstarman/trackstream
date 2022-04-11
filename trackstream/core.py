# -*- coding: utf-8 -*-

"""Stream track fitter and fit result."""

##############################################################################
# IMPORTS

from __future__ import annotations
from optparse import Option

# STDLIB
from typing import Any, Callable, Dict, Optional, Tuple, TypedDict, cast

# THIRD PARTY
import astropy.units as u
from astropy.coordinates import BaseCoordinateFrame, CartesianRepresentation, SkyCoord
from astropy.units import Quantity
from astropy.utils.metadata import MetaAttribute, MetaData
from interpolated_coordinates import InterpolatedSkyCoord
from numpy import array, ndarray
from scipy.linalg import block_diag
from astropy.utils.misc import indent

# LOCAL
from trackstream.kalman.core import FirstOrderNewtonianKalmanFilter, kalman_output
from trackstream.kalman.helper import make_R
from trackstream.rotated_frame import FrameOptimizeResult, RotatedFrameFitter
from trackstream.som import SelfOrganizingMap1D
from trackstream.utils.path import Path, concatenate_paths, path_moments

from .base import CommonBase

__all__ = ["TrackStream", "StreamTrack"]

##############################################################################
# CODE
##############################################################################


class _TrackStreamCachedDict(TypedDict):
    frame: Optional[BaseCoordinateFrame]
    frame_fit: Optional[FrameOptimizeResult]
    frame_fitter: Optional[RotatedFrameFitter]
    arm1_visit_order: Optional[ndarray]
    arm1_SOM: Optional[SelfOrganizingMap1D]
    arm1_mean_path: Optional[kalman_output]
    arm1_kalman: Optional[FirstOrderNewtonianKalmanFilter]
    arm2_visit_order: Optional[ndarray]
    arm2_SOM: Optional[SelfOrganizingMap1D]
    arm2_mean_path: Optional[kalman_output]
    arm2_kalman: Optional[FirstOrderNewtonianKalmanFilter]


class TrackStream:
    """Track a Stream.

    When run, produces a StreamTrack.

    Parameters
    ----------
    onsky : bool or None, optional
        Should the track be fit by on-sky or with distances.
        If None (default) the data is inspected to see if it has distances.

    arm1SOM, arm2SOM : `~trackstream.SelfOrganizingMap1D` or None (optional, keyword-only)
        Fiducial SOMs for stream arms 1 and 2, respectively.
        Warning: the attribute ``onsky`` will be mutated in-place to `onsky`.
    """

    _onsky: Optional[bool]

    def __init__(
        self,
        onsky: Optional[bool],
        *,
        arm1SOM: Optional[SelfOrganizingMap1D] = None,
        arm2SOM: Optional[SelfOrganizingMap1D] = None,
    ) -> None:
        self._onsky = onsky  # mutable

        # SOM
        if arm1SOM is not None:
            arm1SOM.onsky = onsky
        self._arm1_SOM = arm1SOM

        if arm2SOM is not None:
            arm2SOM.onsky = onsky
        self._arm2_SOM = arm2SOM

        self._cache = _TrackStreamCachedDict(
            frame=None,
            frame_fit=None,
            frame_fitter=None,
            arm1_visit_order=None,
            arm1_SOM=arm1SOM,
            arm1_mean_path=None,
            arm1_kalman=None,
            arm2_visit_order=None,
            arm2_SOM=arm2SOM,
            arm2_mean_path=None,
            arm2_kalman=None,
        )

    @property
    def onsky(self) -> Optional[bool]:
        """Whether to fit on-sky or 3d."""
        return self._onsky

    @onsky.setter
    def onsky(self, value: bool) -> None:
        self._onsky = value

    def __repr__(self) -> str:
        """String representation."""
        r = ""

        # 1) header (standard repr)
        header: str = object.__repr__(self)
        r += header

        # 2) on-sky
        r += "\n  on-sky: " + str(self.onsky)

        return r

    # ===============================================================
    # Fit

    def _fit_rotated_frame(
        self,
        stream: "Stream",
        /,
        rot0: Optional[Quantity] = Quantity(0, u.deg),
        bounds: Optional[ndarray] = None,
        **kwargs: Any,
    ) -> Tuple[BaseCoordinateFrame, FrameOptimizeResult]:
        """Fit a rotated frame in `astropy.coordinates.ICRS` coordinates.

        Parameters
        ----------
        stream : `trackstream.stream.Stream`, positional-only
        rot0 : |Quantity| or None, optional
            Initial guess for rotation.
        bounds : array-like or None, optional
            Parameter bounds. If `None`, these are automatically constructed.
            If provided these are used over any other bounds-related arguments.
            ::
                [[rot_low, rot_up],
                 [lon_low, lon_up],
                 [lat_low, lat_up]]

        Other Parameters
        ----------------
        rot_lower, rot_upper : |Quantity|, (optional, keyword-only)
            The lower and upper bounds in degrees.
            Default is (-180, 180] degree.
        origin_lim : |Quantity|, (optional, keyword-only)
            The symmetric lower and upper bounds on origin in degrees.
            Default is 0.005 degree.

        fix_origin : bool or None (optional, keyword-only)
            Whether to fix the origin point. Default is False.
        leastsquares : bool or None (optional, keyword-only)
            Whether to to use :func:`~scipy.optimize.least_square` or
            :func:`~scipy.optimize.minimize`. Default is `False`.

        align_v : bool or None (optional, keyword-only)
            Whether to align velocity to be in positive direction

        Returns
        -------
        BaseCoordinateFrame
        FrameOptimizeResult

        Raises
        ------
        TypeError
            If ``_data_frame`` is None
        """
        # Make and run fitter, starting from the original coordinates
        fitter = RotatedFrameFitter(origin=stream.origin, frame=stream.data_frame, **kwargs)
        fitted = fitter.fit(stream.data_coords, rot0=rot0, bounds=bounds)

        # Cache and return results
        self._cache["frame"] = fitted.frame  # SkyOffsetICRS
        self._cache["frame_fit"] = fitted
        self._cache["frame_fitter"] = fitter

        return fitted.frame, fitted

    # -------------------------------------------

    def _fit_SOM(
        self,
        arm: SkyCoord,
        onsky: bool,
        som: Optional[SelfOrganizingMap1D] = None,
        *,
        learning_rate: float = 0.1,
        sigma: float = 1.0,
        iterations: int = 10_000,
        random_seed: Optional[int] = None,
        progress: bool = False,
        nlattice: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[ndarray, SelfOrganizingMap1D]:
        """Reorder data by SOM.

        Parameters
        ----------
        arm : SkyCoord
        som : object or None, optional
            The self-organizing map. If None, will be constructed.
        learning_rate : float, optional keyword-only
        sigma : float, optional keyword-only
        iterations : int, optional keyword-only

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
            # determine number of lattice points, with a minimum of 5
            nlattice = max(len(arm) // 10, 5) if nlattice is None else nlattice

            som = SelfOrganizingMap1D(
                nlattice,
                onsky,
                frame=arm.frame.replicate_without_data(),
                representation_type=arm.representation_type,  # TODO? as arg
                sigma=sigma,
                learning_rate=learning_rate,
                neighborhood_function="gaussian",
                activation_distance="euclidean",
                random_seed=random_seed,
            )

            # call method to initialize SOM weights
            som.init_prototypes_binned(arm, onsky=onsky, **kwargs)

        # get the ordering by "vote" of the Prototypes
        visit_order = som.fit_predict(
            arm,
            num_iteration=iterations,
            random_order=False,
            progress=progress,
            origin=arm.origin,
        )

        # TODO! optionally transition matrix, i.e. iterative training

        return visit_order, som

    def _fit_kalman_filter(
        self, data: SkyCoord, onsky: bool, **options: Any
    ) -> Tuple[kalman_output, FirstOrderNewtonianKalmanFilter, Path]:
        """Fit data with Kalman filter.

        Parameters
        ----------
        data : SkyCoord
            In frame ``frame`` with ``representation_type``
        frame : BaseCoordinateFrame
        w0 : array or None, optional
            The starting point of the Kalman filter.

        Returns
        -------
        mean_path
        kalman_filter
        """
        # TODO! detect if want positions only, or also velocities
        #       currently only does the positions
        ndims = 3  # no differentials

        # TODO! check data compatible with onsky
        representation_type = data.representation_type
        data = SkyCoord(data.realize_frame(data.data.without_differentials()), copy=False)
        data.representation_type = representation_type

        frame = data.frame.replicate_without_data()
        frame.representation_type = representation_type

        # starting point
        x0 = options.get("x0", None)
        if x0 is None:  # need to determine a good starting point
            # Instead of choosing the first point as the starting point,
            # since the stream is in its frame, instead choose the locus of
            # points near the origin.
            x0 = data.cartesian[:3].mean()  # fist point

        # TODO! as options
        p = array([[0.0001, 0], [0, 1]])
        P0 = block_diag(*(p,) * ndims)

        R0 = make_R(array([0.05] * ndims))[0]  # TODO! actual errors
        options["q_kw"] = dict(var=0.01, n_dims=ndims)  # TODO! overridable

        kf = FirstOrderNewtonianKalmanFilter(
            x0,
            P0,
            R0=R0,
            frame=frame,
            representation_type=CartesianRepresentation,
            **options,  # TODO! from data
        )

        if onsky:
            dt0 = Quantity(0.5, u.deg)
            vmin = Quantity(0.01, u.deg)
        else:
            dt0 = Quantity(50, u.pc)
            vmin = Quantity(0.01, u.pc)
        timesteps = kf.make_simple_timesteps(data, dt0=dt0, width=6, vmin=vmin)

        mean_path, path = kf.fit(data, timesteps=timesteps, representation_type=representation_type)

        return mean_path, kf, path

    # -------------------------------------------

    def fit(
        self,
        stream: "Stream",  # type: ignore
        /,
        onsky: Optional[bool] = None,
        *,
        tune_SOM: bool = True,
        rotated_frame_fit_kw: Optional[dict] = None,
        som_fit_kw: Optional[dict] = None,
        kalman_fit_kw: Optional[dict] = None,
    ) -> StreamTrack:
        """Fit a data to the data.

        Parameters
        ----------
        stream : `trackstream.Stream`, positional-only
        onsky : bool or None, optional
            Should the track be fit by on-sky or with distances.
            If None (default) the data is inspected to see if it has distances.

        tune_SOM : bool, optional keyword-only
        rotated_frame_fit_kw : dict or None, optional keyword-only
        som_fit_kw : dict or None, optional keyword-only
        kalman_fit_kw : dict or None, optional keyword-only

        Returns
        -------
        StreamTrack instance
            Also stores as ``.track``
        """
        onsky = self.onsky if onsky is None else onsky
        if onsky is None:  # ie self.onsky is None
            onsky = not stream.has_distances
        elif not onsky and not stream.has_distances:
            raise ValueError(
                "This stream does not have distance information; cannot compute 3d track."
            )

        # --------------------------------------
        # Fit Rotated Frame
        # this step applies to all arms. In fact, it will perform better if
        # both arms are present, limiting the influence of the tails on the
        # frame orientation.

        frame: BaseCoordinateFrame
        _frame: Optional[BaseCoordinateFrame]
        frame_fit: Optional[FrameOptimizeResult] = None

        # 1) Already provided or in cache.
        #    Either way, don't need to repeat the process.
        _frame = stream.system_frame  # NOT .system_frame ?

        # 2) Fit (& cache), if still None.
        if _frame is None:
            kw: dict = rotated_frame_fit_kw or {}
            frame = stream.fit_frame(**kw)
        else:
            frame = _frame

        # get arms, in frame
        # done after caching b/c coords can use the cache.
        # transforming just to explicitly show it's in the frame
        arm1: SkyCoord = stream.arm1.coords.transform_to(frame)
        arm1.representation_type = frame.representation_type
        arm2: SkyCoord = stream.arm2.coords.transform_to(frame)
        arm2.representation_type = frame.representation_type

        # --------------------------------------
        # Self-Organizing Map
        # Unlike the previous step, we must do this for both arms.

        som_kw = som_fit_kw or {}  # None -> dict
        som_kw.setdefault("maxsep", Quantity(5, u.deg) if onsky else None)

        # --------------------
        # Arm 1

        som1 = self._arm1_SOM
        visit_order1 = None

        # 1) Try to get from cache
        if som1 is None:
            print("SOM is None")
            visit_order1 = self._cache.get("arm1_visit_order", None)
            som1 = self._cache.get("arm1_SOM", None)

        # 2) Fit, if None or doing further tuning
        if som1 is None:
            print("SOM is still None. kwargs are:", som_fit_kw)
            visit_order1, som1 = self._fit_SOM(arm1, onsky, som=som1, **som_kw)
        else:
            if som1.onsky != onsky:  # first need to check consistency of fit type.
                raise TypeError
            if som1.frame != frame:
                raise ValueError

            if tune_SOM:  # TODO! incorporate existing visit_order1
                print("SOM is tuning. kwargs are:", som_fit_kw)
                visit_order1, som1 = self._fit_SOM(arm1, onsky, som=som1, **som_kw)

        # 3) if it's still None, give up
        if visit_order1 is None:
            raise ValueError()

        # cache (even if None)
        self._cache["arm1_visit_order"] = visit_order1
        self._cache["arm1_SOM"] = som1

        arm1 = cast(SkyCoord, arm1[visit_order1])  # re-order

        # --------------------
        # Arm 2 (if not None)

        som2 = self._arm2_SOM
        visit_order2 = None

        if stream.arm2.has_data:

            # 1) try to get from cache (e.g. first time fitting)
            if som2 is None:
                print("SOM is None")
                visit_order2 = self._cache.get("arm2_visit_order", None)
                som2 = self._cache.get("arm2_SOM", None)

            # 2) fit, if still None or force continued fit
            if som2 is None:
                print("SOM is still None. kwargs are:", som_fit_kw)
                visit_order2, som2 = self._fit_SOM(arm2, onsky, som=som2, **som_kw)
            else:
                if som2.onsky != onsky:  # first need to check consistency of fit type.
                    raise TypeError
                if som2.frame != frame:
                    raise ValueError

                if tune_SOM:
                    print("SOM is tuning. kwargs are:", som_fit_kw)
                    # arm2 = arm2[visit_order2]  # TODO!
                    visit_order2, som2 = self._fit_SOM(arm2, onsky, som=som2, **som_kw)

            # 3) if it's still None, give up
            if visit_order2 is None:
                raise ValueError

        # cache (even if None)
        self._cache["arm2_visit_order"] = visit_order2
        self._cache["arm2_SOM"] = som2

        arm2 = cast(SkyCoord, arm2[visit_order2])  # re-order

        # -------------------
        # Kalman Filter
        # both arms start at 0 displacement wrt themselves, but not each other.
        # e.g. the progenitor is cut out. To address this the start of affine
        # is offset by epsilon = min(1e-10, 1e-10 * dp2p[0])

        # Arm 1  (never None)
        # -----
        kalman_fit_kw = kalman_fit_kw or {}
        mean1, kf1, path1 = self._fit_kalman_filter(arm1, onsky, **kalman_fit_kw)
        # cache
        self._cache["arm1_mean_path"] = mean1
        self._cache["arm1_kalman"] = kf1

        # Arm 2
        # -----
        if not stream.arm2.has_data:
            mean2 = kf2 = path2 = None
        else:
            mean2, kf2, path2 = self._fit_kalman_filter(arm2, onsky, **kalman_fit_kw)

        # cache (even if None)
        self._cache["arm2_mean_path"] = mean2
        self._cache["arm2_kalman"] = kf2

        # -------------------
        # Combine together into a single path
        # Need to reverse order of one arm to be indexed toward origin, not away

        if path2 is not None:
            path = concatenate_paths((path2, path1))  # TODO! which negative?
        else:
            path = path1

        # construct interpolation
        track = StreamTrack(
            path,
            stream_data=stream.data,
            origin=stream.origin,
            name=stream.full_name,
            # frame=frame,
            # metadata
            frame_fit=frame_fit,
            # visit_order=visit_order,  # TODO! not combined
            som=dict(
                arm1=self._cache.get("arm1_SOM", None),  # TODO! fix ordering
                arm2=self._cache.get("arm2_SOM", None),
            ),
            kalman=dict(arm1=kf1, arm2=kf2),
        )
        return track

    # ===============================================================


#     def predict(self, affine: Quantity) -> StreamTrack:
#         """Predict from a fit.
#
#         Returns
#         -------
#         StreamTrack instance
#
#         """
#         return self.track(affine)
#
#     def fit_predict(self, stream: "Stream", affine: Quantity, **fit_kwargs: Any) -> StreamTrack:
#         """Fit and Predict."""
#         self.fit(stream, **fit_kwargs)
#         return self.predict(affine)


##############################################################################


class StreamTrack(CommonBase):
    """A stream track interpolation as function of arc length.

    The track is Callable, returning a Frame.

    Parameters
    ----------
    path : `~trackstream.utils.path.Path`
    origin
        of the coordinate system (often the progenitor)
    """

    _name: Optional[str]
    _meta: Dict[str, Any]
    meta = MetaData()

    frame_fit = MetaAttribute()
    visit_order = MetaAttribute()
    som = MetaAttribute()
    kalman = MetaAttribute()

    def __init__(
        self, path: Path, origin: SkyCoord, *, name: Optional[str] = None, **meta: Any
    ) -> None:
        super().__init__(frame=path.frame)
        self._name = name

        # validation of types
        if not isinstance(path, Path):
            raise TypeError("`path` must be <Path>.")
        elif not isinstance(origin, (SkyCoord, BaseCoordinateFrame)):
            raise TypeError("`origin` must be <|SkyCoord|, |Frame|>.")

        # assign
        self._path: Path = path
        self._origin = origin

        # set the MetaAttribute(s)
        for attr in list(meta):
            descr = getattr(self.__class__, attr, None)
            if isinstance(descr, MetaAttribute):
                setattr(self, attr, meta.pop(attr))
        # and the meta
        self._meta.update(meta)

    @property
    def name(self) -> Optional[str]:
        """Return stream-track name."""
        return self._name

    @property
    def path(self) -> Path:
        return self._path

    @property
    def track(self) -> InterpolatedSkyCoord:
        """The path's central track."""
        return self._path.data

    @property
    def affine(self) -> Quantity:
        return self._path.affine

    @property
    def origin(self) -> SkyCoord:
        return self._origin

    #######################################################
    # Math on the Track

    def __call__(self, affine: Optional[Quantity] = None, *, angular: bool = False) -> path_moments:
        """Get discrete points along interpolated stream track.

        Parameters
        ----------
        affine : `~astropy.units.Quantity` array-like or None, optional
            The affine interpolation parameter. If None (default), return
            path moments evaluated at all "tick" interpolation points.
        angular : bool, optional keyword-only
            Whether to compute on-sky or real-space.

        Returns
        -------
        `trackstream.utils.path.path_moments`
            Realized from the ``.path`` attribute.
        """
        return self.path(affine=affine, angular=angular)

    def probability(
        self,
        point: SkyCoord,
        background_model: Optional[Callable[[SkyCoord], Quantity[u.percent]]] = None,
        *,
        angular: bool = False,
        affine: Optional[Quantity] = None,
    ) -> Quantity[u.percent]:
        """Probability point is part of the stream.

        .. todo:: angular probability

        """
        # # Background probability
        # Pb = background_model(point) if background_model is not None else 0.0
        #
        # #
        # angular = False  # TODO: angular probability
        # afn = self._path.closest_affine_to_point(point, angular=False, affine=affine)
        # pt_w = getattr(self._path, "width_angular" if angular else "width")(afn)
        # sep = getattr(self._path, "separation" if angular else "separation_3d")(
        #     point,
        #     interpolate=False,
        #     affine=afn,
        # )

        # stats.norm.pdf(ps.separation_3d(point))  # FIXME! dimensionality
        raise NotImplementedError("TODO!")

    #######################################################
    # misc

    def __repr__(self) -> str:
        """String representation."""
        rs = []

        # 0) header (standard repr)
        header: str = object.__repr__(self)
        frame_name = self.frame.__class__.__name__
        rep_name = self.track.representation_type.__name__
        header = header.replace("StreamTrack", f"StreamTrack ({frame_name}|{rep_name})")
        rs.append(header)

        # 1) name
        name = str(self.name)
        rs.append("  Name: " + name)

        # 2) data
        rs.append(indent(repr(self._path.data), width=2))

        return "\n".join(rs)


# LOCAL
# This is to solve the circular dependency in type hint forward references
# isort: skip
from trackstream.stream import Stream  # noqa: E402
